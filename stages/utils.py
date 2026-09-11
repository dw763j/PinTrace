import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from loguru import logger
from tqdm import tqdm


def get_pypi_versions(package_name: str, out_dir: str | None = None) -> list[str]:
    """Fetch all versions of a package from PyPI and write ``out_dir/pypi#<name>.json``."""
    if out_dir is None:
        from paths import PYPI_INFO
        out_dir = str(PYPI_INFO)
    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"pypi#{package_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Fetched {package_name} from PyPI: {len(data['releases'])} versions")
        return list(data["releases"].keys())
    logger.warning(f"Failed to fetch {package_name} from PyPI: {response.status_code}")
    return []


def _pypi_fetch_state_path(source_id: str) -> str:
    """Stable state file for a given input, used to resume interrupted fetches."""
    from paths import GLOBAL_CACHE, ensure_dirs
    ensure_dirs()
    key = hashlib.sha256(source_id.encode()).hexdigest()[:16]
    return str(GLOBAL_CACHE / f"pypi_fetch_state_{key}.json")


def _load_versions_from_pypi_info(package_name: str, out_dir: str | None = None) -> list[str]:
    """Read versions from an existing ``out_dir/pypi#<name>.json``; return [] if missing or invalid."""
    if out_dir is None:
        from paths import PYPI_INFO
        out_dir = str(PYPI_INFO)
    path = os.path.join(out_dir, f"pypi#{package_name}.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return list(data.get("releases", {}).keys())
    except Exception:
        return []


def parse_top_pypi_package_names(
    path_or_data: str | dict,
    *,
    limit: int | None = None,
) -> list[str]:
    """Parse package names from a ``top-pypi-packages-30-days.min.json`` payload.

    ``path_or_data`` may be a JSON path or an already parsed dict with ``rows``,
    each containing ``project``.
    """
    if isinstance(path_or_data, str):
        with open(path_or_data, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = path_or_data
    rows = data.get("rows", [])
    names = [r["project"] for r in rows if isinstance(r, dict) and "project" in r]
    if limit is not None:
        names = names[:limit]
    return names


def fetch_pypi_versions_parallel(
    names: list[str],
    *,
    out_dir: str | None = None,
    max_workers: int = 16,
    resume: bool = True,
    source_id: str | None = None,
) -> dict[str, list[str]]:
    """Fetch PyPI versions for each name in parallel; write files under ``out_dir``.

    Returns a mapping from package name to version list.

    - ``out_dir`` defaults to ``paths.PYPI_INFO``.
    - Progress is shown with tqdm.
    - When ``resume`` is true, completed names are loaded from a state file keyed
      by ``source_id`` (default: a hash of ``names``) and ``out_dir``. Re-run with
      the same ``names`` and ``out_dir`` to continue after an interrupt.
    """
    if not names:
        return {}
    if out_dir is None:
        from paths import PYPI_INFO
        out_dir = str(PYPI_INFO)
    base_sid = source_id if source_id is not None else "names:" + hashlib.sha256(json.dumps(names, sort_keys=True).encode()).hexdigest()
    sid = f"{base_sid}|{out_dir}"

    result: dict[str, list[str]] = {}
    state_path = _pypi_fetch_state_path(sid)
    completed: set[str] = set()

    if resume and os.path.exists(state_path):
        try:
            with open(state_path, encoding="utf-8") as f:
                state = json.load(f)
            if state.get("source_id") == sid:
                completed = set(state.get("completed", []))
                for n in completed:
                    result[n] = _load_versions_from_pypi_info(n, out_dir)
                logger.info(f"Resume: {len(completed)}/{len(names)} already done")
        except Exception:
            pass

    names_todo = [n for n in names if n not in completed]
    if not names_todo:
        return result

    def _save_state(done: set[str]) -> None:
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({"source_id": sid, "completed": sorted(done)}, f, ensure_ascii=False, indent=0)

    def _fetch(name: str) -> list[str]:
        return get_pypi_versions(name, out_dir=out_dir)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {executor.submit(_fetch, n): n for n in names_todo}
        done = set(completed)
        with tqdm(total=len(names), initial=len(completed), unit="pkg", desc="PyPI fetch") as pbar:
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    result[name] = future.result()
                except Exception:
                    result[name] = []
                done.add(name)
                _save_state(done)
                pbar.update(1)

    return result


def load_mapping(url: str = "https://raw.githubusercontent.com/bndr/pipreqs/refs/heads/master/pipreqs/mapping") -> tuple[dict, set[str]]:
    """Load an import-name to PyPI-name mapping from ``url``."""
    from paths import GLOBAL_CACHE, ensure_dirs
    ensure_dirs()
    mapping_path = GLOBAL_CACHE / "mapping.json"
    cache = None
    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
            if 'url' in loaded_data and loaded_data['url'] == url:
                return loaded_data['mapping'], loaded_data['repeated_import_names']
            cache = loaded_data['mapping']
    response = requests.get(url)
    results = {}
    repeated_import_names = set()
    for line in response.text.splitlines():
        import_name, pypi_name = line.strip().split(":")
        if import_name in results:
            results[import_name] = pypi_name.lower()
            repeated_import_names.add(import_name)
        else:
            results[import_name] = pypi_name.lower()
    # Merge the previous cache into the freshly fetched mapping.
    if cache:
        results.update(cache)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({'url': url, 'repeated_import_names': list(repeated_import_names), 'mapping': results}, f, ensure_ascii=False, indent=2)
    return results, repeated_import_names

PACKAGE_ALIAS_OVERRIDES = {
    # django-ckeditor exposes this top-level module but pipreqs' upstream
    # mapping does not currently include it.
    "ckeditor_uploader": "django-ckeditor",
}


def get_pkg_name(pkg: str, mapping: dict | None = None) -> str:
    if mapping is None:
        mapping, _ = load_mapping()
    return PACKAGE_ALIAS_OVERRIDES.get(pkg, mapping.get(pkg, pkg)).lower()

def extract_code_from_content(content: str) -> str:
    if "<think>" in content and "</think>" in content:
        content = content.split("</think>")[1].strip()
    if content.startswith("```python") and content.endswith("```"):
        content = content[len("```python"):-len("```")]
    elif content.startswith("```") and content.endswith("```"):
        content = content[len("```"):-len("```")]
    return content.strip()

def get_vul_tpls(vul_tpls_path: str) -> dict[str, list[str]]:
    vul_tpls = {}
    for vul in os.listdir(vul_tpls_path):
        try:
            with open(os.path.join(vul_tpls_path, vul), 'r') as f:
                data = json.load(f)
                if not vul_tpls.get(data['affected'][0]['package']['name'].lower(), []):
                    vul_tpls[data['affected'][0]['package']['name'].lower()] = []
                versions = data['affected'][0].get('versions', [])
                vul_tpls[data['affected'][0]['package']['name'].lower()].extend(versions)
        except Exception:
            logger.error(f"Failed to load {vul} from {vul_tpls_path}")
            raise
    return vul_tpls

if __name__ == "__main__":
    from paths import OSV_INDEX, RESOURCES, LOGS, ensure_dirs
    ensure_dirs()
    logger.add(str(LOGS / "fetch_pypi_vul_tpls_versions_parallel.log"))
    vul_tpls = get_vul_tpls(str(OSV_INDEX))
    names = list(set(vul_tpls.keys()))
    fetch_pypi_versions_parallel(names, out_dir=str(RESOURCES / "pypi_info_vul_tpls"))

# get_pypi_versions("bleach")
# extract_code_from_`content("```python\nprint('Hello, World!')\n```")
# extract_code_from_content("print('Hello, World!')\n")
# extract_code_from_content("```\nprint('Hello, World!')\n```")
# extract_code_from_content("```python\nprint('Hello, World!')\n```")
# extract_code_from_content("```python\nprint('Hello, World!')\n```")

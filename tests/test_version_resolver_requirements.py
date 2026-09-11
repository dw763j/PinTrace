"""Test requirements.txt extraction and version resolution (including yanked) in version_resolver."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stages.version_resolver import (
    UvResolverConfig,
    _build_release_time_index,
    _extract_code_blocks_by_lang,
    _is_version_yanked,
    _merge_requirements_and_code,
    _parse_requirements_txt,
    build_task_requirements,
    extract_tpl_versions_from_llm_output,
    parse_uv_lock_pins,
    resolve_record_with_uv,
    resolve_version,
)


def test_extract_blocks():
    content = """```requirements.txt
cryptography
```

```python
import base64
from cryptography.fernet import Fernet
def task_func():
    pass
```"""
    blocks = _extract_code_blocks_by_lang(content)
    assert "requirements.txt" in blocks
    assert blocks["requirements.txt"].strip() == "cryptography"
    assert "python" in blocks
    assert "cryptography" in blocks["python"]


def test_extract_blocks_reversed():
    content = """```python
import requests
```
```requirements.txt
requests
```"""
    blocks = _extract_code_blocks_by_lang(content)
    assert blocks["python"].strip() == "import requests"
    assert blocks["requirements.txt"].strip() == "requests"


def test_parse_requirements():
    text = "cryptography\npandas==2.0.0\nnumpy>=1.20\n# comment\nrequests  # http\nscipy>=1.0,<2.0"
    items = _parse_requirements_txt(text)
    assert len(items) == 5
    by_lib = {x["library"]: x for x in items}
    assert by_lib["cryptography"]["version"] is None and by_lib["cryptography"]["version_spec"] is None
    assert by_lib["pandas"]["version"] == "2.0.0" and by_lib["pandas"]["version_spec"] is None
    assert by_lib["numpy"]["version"] is None and by_lib["numpy"]["version_spec"] == ">=1.20"
    assert by_lib["requests"]["version"] is None and by_lib["requests"]["version_spec"] is None
    assert set(by_lib["scipy"]["version_spec"].split(",")) == {">=1.0", "<2.0"}


def test_parse_pep508_requirements_and_origin():
    items = _parse_requirements_txt(
        'numpy~=1.26\nfoo[cli]>=1; python_version < "3.13"\n'
        'demo @ https://example.com/demo-1.0-py3-none-any.whl\ninvalid requirement !!!'
    )
    numpy = items[0]
    assert numpy["version"] is None
    assert numpy["version_spec"] == "~=1.26"
    assert numpy["constraint_kind"] == "range"
    assert numpy["declaration_source"] == "manifest"
    assert items[1]["extras"] == ["cli"]
    assert items[1]["marker"] == 'python_version < "3.13"'
    assert items[2]["direct_url"].startswith("https://")
    assert items[2]["constraint_kind"] == "range"
    assert items[3]["constraint_kind"] == "invalid"
    assert items[3]["parse_error"]


def test_inline_origin_for_pinned_and_bare_imports():
    from stages.version_resolver import _extract_tpl_versions_from_code

    items = _extract_tpl_versions_from_code(
        "import requests  # VERSION=2.31.0\nimport pandas", "3.12"
    )
    by_lib = {item["library"]: item for item in items}
    assert by_lib["requests"]["declaration_source"] == "inline_annotation"
    assert by_lib["requests"]["constraint_kind"] == "exact"
    assert by_lib["pandas"]["declaration_source"] == "code_import_only"
    assert by_lib["pandas"]["constraint_kind"] == "bare"


def test_ast_import_extraction_handles_multiple_aliases_and_multiline_imports():
    from stages.version_resolver import _extract_tpl_versions_from_code

    items = _extract_tpl_versions_from_code(
        """import os, requests, numpy as np
from pandas.core import (
    frame,
)  # VERSION=2.2.0
from .local import helper
""",
        "3.12",
    )
    by_lib = {item["library"]: item for item in items}
    assert set(by_lib) == {"requests", "numpy", "pandas"}
    assert by_lib["requests"]["constraint_kind"] == "bare"
    assert by_lib["numpy"]["constraint_kind"] == "bare"
    assert by_lib["pandas"]["version"] == "2.2.0"
    assert by_lib["pandas"]["declaration_source"] == "inline_annotation"


def test_build_task_requirements_preserves_model_constraints():
    items = _parse_requirements_txt('requests==2.31.0\nnumpy~=1.26\nfoo[cli]; python_version < "3.13"')
    assert build_task_requirements(items, {}) == [
        "requests==2.31.0",
        "numpy~=1.26",
        'foo[cli]; python_version < "3.13"',
    ]


def test_parse_uv_lock_pins_with_hash_continuations():
    lock = """requests==2.31.0 \\
    --hash=sha256:abc
urllib3==2.2.3
"""
    assert parse_uv_lock_pins(lock) == {"requests": "2.31.0", "urllib3": "2.2.3"}


def test_uv_resolution_propagates_origin_and_transitives(tmp_path):
    record = {
        "task_id": "task/1",
        "extracted_versions": _parse_requirements_txt("requests==2.31.0\nnumpy>=1.20\npandas"),
    }

    def fake_run(command, **kwargs):
        lock_path = Path(command[command.index("--output-file") + 1])
        lock_path.write_text(
            "requests==2.31.0\nnumpy==1.26.4\npandas==2.2.0\nurllib3==2.2.3\n",
            encoding="utf-8",
        )
        return type("Proc", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    with patch("stages.version_resolver.subprocess.run", side_effect=fake_run):
        resolved = resolve_record_with_uv(
            record,
            config=UvResolverConfig(uv_bin="uv"),
            lock_root=tmp_path,
            mapping={},
        )
    by_name = {x["pypi_name"]: x for x in resolved["per_lib"]}
    assert by_name["requests"]["resolution_origin"] == "model_exact"
    assert by_name["numpy"]["resolution_origin"] == "model_constrained_resolver"
    assert by_name["pandas"]["resolution_origin"] == "unconstrained_resolver"
    transitive = [x for x in resolved["resolved_environment"] if not x["is_direct"]]
    assert [(x["pypi_name"], x["resolved_version"]) for x in transitive] == [("urllib3", "2.2.3")]


def test_uv_resolution_preserves_invalid_declaration_without_invoking_uv(tmp_path):
    invalid = _parse_requirements_txt("not a valid requirement !!!")
    with patch("stages.version_resolver.subprocess.run") as mocked:
        resolved = resolve_record_with_uv(
            {"task_id": "invalid", "extracted_versions": invalid},
            config=UvResolverConfig(),
            lock_root=tmp_path,
            mapping={},
        )
    mocked.assert_not_called()
    assert resolved["resolver"]["status"] == "empty"
    assert resolved["per_lib"][0]["constraint_kind"] == "invalid"
    assert resolved["per_lib"][0]["resolution_origin"] == "unresolved"


def test_task_failure_validates_model_constraint_independently(tmp_path):
    items = _parse_requirements_txt("requests>=2\nnot-a-real-package")

    def fake_run(command, **kwargs):
        input_path = Path(command[3])
        output_path = Path(command[command.index("--output-file") + 1])
        if input_path.name.startswith("constraint-"):
            output_path.write_text("requests==2.31.0\n", encoding="utf-8")
            return type("Proc", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        return type("Proc", (), {"returncode": 1, "stdout": "", "stderr": "no solution"})()

    with patch("stages.version_resolver.subprocess.run", side_effect=fake_run):
        resolved = resolve_record_with_uv(
            {"task_id": "conflict", "extracted_versions": items},
            config=UvResolverConfig(),
            lock_root=tmp_path,
            mapping={},
        )
    request = next(lib for lib in resolved["per_lib"] if lib["pypi_name"] == "requests")
    assert resolved["resolver"]["status"] == "failed"
    assert request["resolved_version"] is None
    assert request["constraint_valid"] is True
    assert request["constraint_resolved_version"] == "2.31.0"
    assert request["resolution_origin"] == "model_constrained_resolver"


def test_uv_timeout_with_bytes_output_is_json_serializable(tmp_path):
    items = _parse_requirements_txt("requests==2.31.0")

    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=command,
            timeout=kwargs.get("timeout") or 1,
            output=b"partial stdout \xff",
            stderr=b"uv still compiling",
        )

    with patch("stages.version_resolver.subprocess.run", side_effect=fake_run):
        resolved = resolve_record_with_uv(
            {"task_id": "timeout", "extracted_versions": items},
            config=UvResolverConfig(),
            lock_root=tmp_path,
            mapping={},
            timeout_seconds=1,
        )

    json.dumps(resolved)
    assert resolved["resolver"]["status"] == "failed"
    assert resolved["resolver"]["returncode"] == 124
    assert isinstance(resolved["resolver"]["stdout"], str)
    assert isinstance(resolved["resolver"]["stderr"], str)
    assert "partial stdout" in resolved["resolver"]["stdout"]
    assert "uv still compiling" in resolved["resolver"]["stderr"]


def test_merge_req_and_code():
    req = _parse_requirements_txt("cryptography")
    code = [
        {"library": "cryptography", "version": None, "import_line": "from cryptography.fernet import Fernet"},
        {"library": "base64", "version": None, "import_line": "import base64"},
    ]
    mapping = {}
    merged = _merge_requirements_and_code(req, code, "3.12", mapping)
    names = [m["library"] for m in merged]
    assert "cryptography" in names
    assert "base64" not in names  # stdlib filtered
    cryptography = next(m for m in merged if m["library"] == "cryptography")
    assert cryptography["declaration_source"] == "manifest"
    assert cryptography["constraint_kind"] == "bare"


def test_merge_preserves_conflicting_duplicate_constraints_for_uv():
    from stages.version_resolver import _extract_tpl_versions_from_code

    req = _parse_requirements_txt("requests<2\nrequests>=3")
    code = _extract_tpl_versions_from_code("import requests", "3.12")
    merged = _merge_requirements_and_code(req, code, "3.12", {})
    assert build_task_requirements(merged, {}) == ["requests<2", "requests>=3"]


def test_full_extraction():
    sample_path = Path(__file__).parents[1] / "outputs/d1/requirements.txt/pipeline_gemini-3-pro-preview_test_10/gemini-3-pro-preview_requirements.txt.jsonl"
    if not sample_path.exists():
        return
    records, _ = extract_tpl_versions_from_llm_output(str(sample_path))
    assert len(records) > 0
    r0 = records[0]
    assert "extracted_versions" in r0
    assert len(r0["extracted_versions"]) > 0
    assert r0["extracted_versions"][0]["library"] in ("cryptography", "psutil", "requests", "pandas", "numpy")


def test_is_version_yanked():
    releases = {
        "1.0": [{"yanked": False, "upload_time_iso_8601": "2020-01-01T00:00:00Z"}],
        "2.0": [{"yanked": True, "upload_time_iso_8601": "2021-01-01T00:00:00Z"}],
        "3.0": [
            {"yanked": False, "upload_time_iso_8601": "2022-01-01T00:00:00Z"},
            {"yanked": True, "upload_time_iso_8601": "2022-01-01T00:00:00Z"},
        ],
    }
    assert _is_version_yanked(releases, "1.0") is False
    assert _is_version_yanked(releases, "2.0") is True
    assert _is_version_yanked(releases, "3.0") is True  # any file yanked => whole release yanked


def test_build_release_time_index_excludes_yanked():
    pypi_json = {
        "releases": {
            "1.0": [{"yanked": False, "upload_time_iso_8601": "2020-01-01T00:00:00Z"}],
            "2.0": [{"yanked": True, "upload_time_iso_8601": "2021-01-01T00:00:00Z"}],
        }
    }
    index = _build_release_time_index(pypi_json, exclude_yanked=True)
    assert "1.0" in index
    assert "2.0" not in index


def test_resolve_version_yanked():
    """When specified version is yanked, version_exists=False, version_status='yanked'."""
    with tempfile.TemporaryDirectory() as tmp:
        pypi_path = os.path.join(tmp, "pypi#yanked-test-pkg.json")
        pypi_json = {
            "info": {"name": "yanked-test-pkg"},
            "releases": {
                "1.0": [{"yanked": False, "upload_time_iso_8601": "2020-01-01T00:00:00Z"}],
                "2.0": [{"yanked": True, "upload_time_iso_8601": "2021-01-01T00:00:00Z"}],
            },
        }
        with open(pypi_path, "w", encoding="utf-8") as f:
            json.dump(pypi_json, f)

        # Need a mapping from import name to pypi name
        mapping = {"yanked_test_pkg": "yanked-test-pkg"}

        r_ok = resolve_version(
            "yanked_test_pkg", "1.0", None, "2024-01-01", pypi_info_dir=tmp, mapping=mapping
        )
        assert r_ok.version_exists is True
        assert r_ok.version_status is None
        assert r_ok.resolved_version == "1.0"

        r_yanked = resolve_version(
            "yanked_test_pkg", "2.0", None, "2024-01-01", pypi_info_dir=tmp, mapping=mapping
        )
        assert r_yanked.version_exists is False
        assert r_yanked.version_status == "yanked"
        assert r_yanked.resolved_version is None


if __name__ == "__main__":
    test_extract_blocks()
    test_extract_blocks_reversed()
    test_parse_requirements()
    test_merge_req_and_code()
    test_full_extraction()
    test_is_version_yanked()
    test_build_release_time_index_excludes_yanked()
    test_resolve_version_yanked()
    print("All tests passed.")

"""Evaluate BigCodeBench samples in per-task isolated uv virtual environments (D1).

Run from project root:
    python -m evaluate.d1.isolated_eval --input-path outputs/<model>.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import GLOBAL_CACHE, OUTPUTS, RESOURCES, ensure_dirs

import argparse
import json
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass

from stdlib_list import stdlib_list

from stages.utils import extract_code_from_content

IMPORT_WITH_VERSION_RE = re.compile(
    r"^\s*import\s+([A-Za-z_][A-Za-z0-9_\.]*)(?:\s+as\s+[A-Za-z_][A-Za-z0-9_]*)?\s*#\s*VERSION\s*=\s*([^\s#]+)\s*$"
)
FROM_IMPORT_WITH_VERSION_RE = re.compile(
    r"^\s*from\s+([A-Za-z_][A-Za-z0-9_\.]*)\s+import\s+.+?\s*#\s*VERSION\s*=\s*([^\s#]+)\s*$"
)


@dataclass
class IsolatedEvalResult:
    task_id: str
    deps_specified: list[str]
    deps_install_ok: list[str]
    deps_install_failed: list[dict[str, str]]
    run_ok: bool
    status: str
    details: dict[str, str]
    error: str | None


def _load_mapping(mapping_path: str) -> dict[str, str]:
    if not Path(mapping_path).exists():
        return {}
    payload = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    raw = payload.get("mapping", payload)
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k.strip().lower()] = v.strip().lower()
    return out


def _top_import_name(name: str) -> str:
    return name.strip().split(".")[0].lower()


def extract_pinned_requirements(
    code: str,
    *,
    mapping: dict[str, str],
    python_version: str,
) -> list[str]:
    stdlibs = set(stdlib_list(python_version))
    specs: list[str] = []
    seen = set()
    for line in code.splitlines():
        m1 = IMPORT_WITH_VERSION_RE.match(line)
        m2 = FROM_IMPORT_WITH_VERSION_RE.match(line)
        match = m1 or m2
        if not match:
            continue
        import_name = _top_import_name(match.group(1))
        version = match.group(2).strip()
        if not version:
            continue
        if import_name in stdlibs:
            continue
        pypi_name = mapping.get(import_name, import_name)
        spec = f"{pypi_name}=={version}"
        if spec not in seen:
            seen.add(spec)
            specs.append(spec)
    return specs


def _load_local_bigcodebench(local_jsonl_path: str) -> dict[str, dict]:
    data = {}
    with open(local_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = row.get("task_id")
            if isinstance(task_id, str):
                data[task_id] = row
    return data


def _load_samples(input_path: str, source_format: str, max_samples: int | None) -> list[dict]:
    rows: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            task_id = obj.get("task_id")
            if not isinstance(task_id, str):
                continue

            if source_format == "llm_output":
                content = ((obj.get("llm_output") or {}).get("content") or "")
                code = extract_code_from_content(content)
            else:
                code = obj.get("solution") or obj.get("completion") or ""
                if source_format == "solution_jsonl" and "completion" in obj and "solution" not in obj:
                    code = obj["completion"]
                code = str(code)
                if "```" in code:
                    code = extract_code_from_content(code)

            rows.append({"task_id": task_id, "code": code})
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def _run_one_in_isolated_env(
    *,
    sample: dict,
    problem: dict,
    mapping: dict[str, str],
    python_version: str,
    project_root: str,
    timeout_seconds: int,
    venv_python: str,
) -> IsolatedEvalResult:
    task_id = sample["task_id"]
    code = sample["code"]
    deps = extract_pinned_requirements(
        code,
        mapping=mapping,
        python_version=python_version,
    )

    install_ok: list[str] = []
    install_failed: list[dict[str, str]] = []

    with tempfile.TemporaryDirectory(prefix=f"bcb_iso_{task_id.replace('/', '_')}_") as tmpdir:
        venv_path = Path(tmpdir) / "venv"
        create = subprocess.run(
            ["uv", "venv", "--python", venv_python, str(venv_path)],
            capture_output=True,
            text=True,
        )
        if create.returncode != 0:
            return IsolatedEvalResult(
                task_id=task_id,
                deps_specified=deps,
                deps_install_ok=[],
                deps_install_failed=[],
                run_ok=False,
                status="env_error",
                details={},
                error=create.stderr[-1200:],
            )

        py = venv_path / "bin" / "python"
        for spec in deps:
            proc = subprocess.run(
                ["uv", "pip", "install", "--python", str(py), spec],
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                install_ok.append(spec)
            else:
                install_failed.append({"spec": spec, "stderr_tail": proc.stderr[-600:]})

        runner = Path(tmpdir) / "runner.py"
        payload = {
            "solution": code,
            "test": problem["test"],
            "entry_point": problem["entry_point"],
            "timeout_seconds": timeout_seconds,
        }
        runner.write_text(
            (
                "import builtins, json, os, sys, time, traceback, types, unittest\n"
                "from pathlib import Path\n"
                "payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))\n"
                "module_name='__test__'\n"
                "mod=types.ModuleType(module_name)\n"
                "mod.__dict__.update({'__builtins__': builtins, '__file__': module_name + '.py', '__package__': None, '__doc__': None, 'sys': sys, 'os': os, 'environ': os.environ})\n"
                "result={'status':'fail','details':{},'error':None}\n"
                "try:\n"
                "    full_code = payload['solution'] + '\\n' + payload['test']\n"
                "    exec(compile(full_code, module_name + '.py', 'exec'), mod.__dict__)\n"
                "    tc = getattr(mod, 'TestCases')\n"
                "    suite = unittest.TestLoader().loadTestsFromTestCase(tc)\n"
                "    tr = unittest.TestResult()\n"
                "    suite.run(tr)\n"
                "    issues = tr.failures + tr.errors\n"
                "    if issues:\n"
                "        for test, trace in issues:\n"
                "            result['details'][test.id().split('.')[-1]] = trace\n"
                "        result['status'] = 'fail'\n"
                "    else:\n"
                "        result['status'] = 'pass'\n"
                "except BaseException:\n"
                "    result['status'] = 'error'\n"
                "    result['error'] = traceback.format_exc()[-3000:]\n"
                "print(json.dumps(result, ensure_ascii=False))\n"
            ),
            encoding="utf-8",
        )
        payload_path = Path(tmpdir) / "payload.json"
        payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        env = os.environ.copy()
        env["PYTHONPATH"] = project_root
        # Match BigCodeBench: run in isolated temp dir (create_tempdir/chdir)
        proc = subprocess.run(
            [str(py), str(runner), str(payload_path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds + 20,
            env=env,
            cwd=str(tmpdir),
        )
        if proc.returncode != 0:
            return IsolatedEvalResult(
                task_id=task_id,
                deps_specified=deps,
                deps_install_ok=install_ok,
                deps_install_failed=install_failed,
                run_ok=False,
                status="runner_error",
                details={},
                error=(proc.stderr or proc.stdout)[-2000:],
            )
        try:
            output = json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            output = {"status": "runner_parse_error", "details": {}, "error": proc.stdout[-2000:]}
        return IsolatedEvalResult(
            task_id=task_id,
            deps_specified=deps,
            deps_install_ok=install_ok,
            deps_install_failed=install_failed,
            run_ok=output.get("status") == "pass",
            status=str(output.get("status")),
            details=output.get("details") or {},
            error=output.get("error"),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BigCodeBench samples in per-task isolated uv virtual environments."
    )
    parser.add_argument("--input-path", required=True, help="Path to samples JSONL (llm output or solution JSONL).")
    parser.add_argument(
        "--source-format",
        choices=["llm_output", "solution_jsonl"],
        default="llm_output",
        help="llm_output: use llm_output.content; solution_jsonl: use solution/completion field.",
    )
    parser.add_argument(
        "--bigcodebench-local-jsonl",
        default=None,
        help="Local BigCodeBench JSONL. Default: resources/BigCodeBench-v0.1.4-local.jsonl",
    )
    parser.add_argument("--mapping-path", default=None, help="Canonical->pypi mapping. Default: global_cache/mapping.json")
    parser.add_argument("--python-version", default="3.12")
    parser.add_argument(
        "--venv-python",
        default="3.12",
        help="Python version used by per-task uv virtual environments (e.g. 3.11).",
    )
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--out-path", default=None, help="Output JSON. Default: outputs/isolated_eval_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    project_root = str(Path(__file__).resolve().parents[2])
    mapping_path = args.mapping_path or str(GLOBAL_CACHE / "mapping.json")
    bigcodebench_path = args.bigcodebench_local_jsonl or str(RESOURCES / "BigCodeBench-v0.1.4-local.jsonl")
    mapping = _load_mapping(mapping_path)
    problems = _load_local_bigcodebench(bigcodebench_path)
    samples = _load_samples(args.input_path, args.source_format, args.max_samples)

    results: list[IsolatedEvalResult] = []
    for sample in samples:
        problem = problems.get(sample["task_id"])
        if not problem:
            results.append(
                IsolatedEvalResult(
                    task_id=sample["task_id"],
                    deps_specified=[],
                    deps_install_ok=[],
                    deps_install_failed=[],
                    run_ok=False,
                    status="missing_task",
                    details={},
                    error=f"task_id not found in local dataset: {sample['task_id']}",
                )
            )
            continue
        one = _run_one_in_isolated_env(
            sample=sample,
            problem=problem,
            mapping=mapping,
            python_version=args.python_version,
            project_root=project_root,
            timeout_seconds=args.timeout_seconds,
            venv_python=args.venv_python,
        )
        results.append(one)
        print(f"[{len(results)}/{len(samples)}] {one.task_id} -> {one.status}")

    out_path = Path(args.out_path) if args.out_path else OUTPUTS / "isolated_eval_results.json"
    out = {
        "input_path": args.input_path,
        "source_format": args.source_format,
        "bigcodebench_local_jsonl": bigcodebench_path,
        "total": len(results),
        "pass": sum(1 for r in results if r.status == "pass"),
        "fail": sum(1 for r in results if r.status == "fail"),
        "error_like": sum(1 for r in results if r.status not in {"pass", "fail"}),
        "results": [asdict(r) for r in results],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

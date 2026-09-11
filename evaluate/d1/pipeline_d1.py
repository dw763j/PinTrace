"""Backward-compatible D1 pipeline entrypoint.

Uses unified implementation in `evaluate.pipeline`.
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end D1 TPL security evaluation pipeline.")
    parser.add_argument("--dataset", default="bigcode/bigcodebench")
    parser.add_argument("--split", default="v0.1.4")
    parser.add_argument("--filter-third-party", action="store_true")
    parser.add_argument("--python-version", default="3.12")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--prompt-mode",
        default="inline",
        choices=["inline", "inline_no_vuln", "requirements.txt", "requirements.txt_no_vuln"],
    )
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-model", default=None)
    parser.add_argument("--api-mode", default="openai", choices=["openai", "openrouter"])
    parser.add_argument("--provider", default=None)  # kept for compatibility, no-op in unified parser
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--reference-date", default="2026-02-01T00:00:00+00:00")
    parser.add_argument("--llm-output-path", default=None)
    parser.add_argument("--outputs-d1", default=None)
    parser.add_argument("--analysis-output-root", default=None)
    parser.add_argument("--resolver", choices=["legacy", "uv"], default="legacy")
    parser.add_argument("--uv-bin", default="uv")
    parser.add_argument("--python-platform", default="x86_64-unknown-linux-gnu")
    parser.add_argument("--exclude-newer", default="2026-02-01T00:00:00Z")
    parser.add_argument("--prerelease", default="if-necessary-or-explicit")
    parser.add_argument("--ty-version", default="0.0.23")
    parser.add_argument("--stages", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--osv-index-path", default=None)
    parser.add_argument("--xdg-cache-path", default=None)
    parser.add_argument("--bigcodebench-path", default=None)
    parser.add_argument("--bcb-timeout", type=int, default=180)
    parser.add_argument("--compat-checkpoint-every", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    argv = [
        "evaluate.pipeline",
        "--track",
        "d1",
        "--dataset",
        args.dataset,
        "--split",
        args.split,
        "--python-version",
        args.python_version,
        "--prompt-mode",
        args.prompt_mode,
        "--api-mode",
        args.api_mode,
        "--workers",
        str(args.workers),
        "--reference-date",
        args.reference_date,
        "--stages",
        *[str(s) for s in args.stages],
        "--bcb-timeout",
        str(args.bcb_timeout),
        "--compat-checkpoint-every",
        str(args.compat_checkpoint_every),
    ]
    if args.run_name:
        argv.extend(["--run-name", args.run_name])
    if args.api_base_url:
        argv.extend(["--api-url", args.api_base_url])
    if args.api_key:
        argv.extend(["--api-key", args.api_key])
    if args.api_model:
        argv.extend(["--api-model", args.api_model])
    if args.max_examples is not None:
        argv.extend(["--max-examples", str(args.max_examples)])
    if args.max_records is not None:
        argv.extend(["--max-records", str(args.max_records)])
    if args.llm_output_path:
        argv.extend(["--llm-output-path", args.llm_output_path])
    if args.outputs_d1:
        argv.extend(["--outputs-d1", args.outputs_d1])
    if args.analysis_output_root:
        argv.extend(["--analysis-output-root", args.analysis_output_root])
    argv.extend([
        "--resolver", args.resolver,
        "--uv-bin", args.uv_bin,
        "--python-platform", args.python_platform,
        "--exclude-newer", args.exclude_newer,
        "--prerelease", args.prerelease,
        "--ty-version", args.ty_version,
    ])
    if args.osv_index_path:
        argv.extend(["--osv-index-path", args.osv_index_path])
    if args.xdg_cache_path:
        argv.extend(["--xdg-cache-path", args.xdg_cache_path])
    if args.bigcodebench_path:
        argv.extend(["--bigcodebench-path", args.bigcodebench_path])
    if args.filter_third_party:
        argv.append("--filter-third-party")

    from evaluate.pipeline import main as unified_main

    old_argv = sys.argv
    try:
        sys.argv = argv
        unified_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()

"""Backward-compatible D2 pipeline entrypoint.

Uses unified implementation in `evaluate.pipeline`.
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end D2 TPL security analysis pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--llm-output-path", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--outputs-d2", default=None)
    parser.add_argument("--stages", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-model", default="deepseek-v3.2")
    parser.add_argument("--api-mode", default="openai", choices=["openai", "openwebui", "openrouter"])
    parser.add_argument("--provider-order", default=None)
    parser.add_argument("--provider-allow-fallbacks", action="store_true")
    parser.add_argument("--http-referer", default=None)
    parser.add_argument("--site-title", default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--prompt-mode", default="blind", choices=["blind", "hint"])
    parser.add_argument(
        "--pinning-mode",
        default="inline",
        choices=["inline", "inline_no_vuln", "requirements.txt"],
    )
    parser.add_argument("--max-hint-chars", type=int, default=1200)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--reference-date", default="2026-02-01T00:00:00+00:00")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--xdg-cache-path", default=None)
    parser.add_argument("--skip-vuln", action="store_true")
    parser.add_argument("--skip-compat", action="store_true")
    parser.add_argument("--python-version", default="3.12")
    parser.add_argument("--analysis-output-root", default=None)
    parser.add_argument("--resolver", choices=["legacy", "uv"], default="legacy")
    parser.add_argument("--uv-bin", default="uv")
    parser.add_argument("--python-platform", default="x86_64-unknown-linux-gnu")
    parser.add_argument("--exclude-newer", default="2026-02-01T00:00:00Z")
    parser.add_argument("--prerelease", default="if-necessary-or-explicit")
    parser.add_argument("--ty-version", default="0.0.23")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    argv = [
        "evaluate.pipeline",
        "--track",
        "d2",
        "--python-version",
        args.python_version,
        "--api-model",
        args.api_model,
        "--api-mode",
        args.api_mode,
        "--timeout",
        str(args.timeout),
        "--prompt-mode",
        args.prompt_mode,
        "--pinning-mode",
        args.pinning_mode,
        "--max-hint-chars",
        str(args.max_hint_chars),
        "--reference-date",
        args.reference_date,
        "--workers",
        str(args.workers),
        "--stages",
        *[str(s) for s in args.stages],
    ]
    if args.dataset:
        argv.extend(["--dataset", args.dataset])
    if args.llm_output_path:
        argv.extend(["--llm-output-path", args.llm_output_path])
    if args.run_name:
        argv.extend(["--run-name", args.run_name])
    if args.outputs_d2:
        argv.extend(["--outputs-d2", args.outputs_d2])
    if args.api_url:
        argv.extend(["--api-url", args.api_url])
    if args.api_key:
        argv.extend(["--api-key", args.api_key])
    if args.provider_order:
        argv.extend(["--provider-order", args.provider_order])
    if args.provider_allow_fallbacks:
        argv.append("--provider-allow-fallbacks")
    if args.http_referer:
        argv.extend(["--http-referer", args.http_referer])
    if args.site_title:
        argv.extend(["--site-title", args.site_title])
    if args.max_examples is not None:
        argv.extend(["--max-examples", str(args.max_examples)])
    if args.max_records is not None:
        argv.extend(["--max-records", str(args.max_records)])
    if args.xdg_cache_path:
        argv.extend(["--xdg-cache-path", args.xdg_cache_path])
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
    if args.skip_vuln:
        argv.append("--skip-vuln")
    if args.skip_compat:
        argv.append("--skip-compat")

    from evaluate.pipeline import main as unified_main

    old_argv = sys.argv
    try:
        sys.argv = argv
        unified_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()

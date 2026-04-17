"""Backward-compatible D2 inference entrypoint.

Uses unified implementation in ``evaluate.inference``.

Usage (repository root; optional: ``source .venv/bin/activate``)::

    python -m evaluate.d2.run_inference --help
"""

from __future__ import annotations

import argparse
import sys
# from pathlib import Path

# from evaluate.inference import run_d2_inference as run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM inference over D2 (StackOverflow) dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-model", default="deepseek-v3.2")
    parser.add_argument("--api-mode", default="openai", choices=["openai", "openwebui", "openrouter"])
    parser.add_argument("--provider-order", default=None)
    parser.add_argument("--provider-allow-fallbacks", action="store_true")
    parser.add_argument("--http-referer", default=None)
    parser.add_argument("--site-title", default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--mode", default="blind", choices=["blind", "hint"])
    parser.add_argument(
        "--pinning-mode",
        default="inline",
        choices=["inline", "inline_no_vuln", "requirements.txt"],
        dest="pinning_mode",
    )
    parser.add_argument("--max-hint-chars", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    argv = [
        "evaluate.inference",
        "--track",
        "d2",
        "--dataset",
        args.dataset,
        "--api-model",
        args.api_model,
        "--api-mode",
        args.api_mode,
        "--timeout",
        str(args.timeout),
        "--prompt-mode",
        args.mode,
        "--pinning-mode",
        args.pinning_mode,
        "--max-hint-chars",
        str(args.max_hint_chars),
        "--workers",
        str(args.workers),
        "--log-level",
        args.log_level,
    ]
    if args.max_examples is not None:
        argv.extend(["--max-examples", str(args.max_examples)])
    if args.output_dir:
        argv.extend(["--output-dir", args.output_dir])
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

    from evaluate.inference import main as unified_main

    old_argv = sys.argv
    try:
        sys.argv = argv
        unified_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()


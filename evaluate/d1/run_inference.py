"""Backward-compatible D1 inference entrypoint.

Uses unified implementation in ``evaluate.inference``.

Usage (repository root; optional: ``source .venv/bin/activate``)::

    python -m evaluate.d1.run_inference --help
"""

from __future__ import annotations

import argparse
import sys

# from evaluate.inference import (
#     LLMConfig,
#     augment_prompt_with_versions,
#     filter_third_party,
#     load_dataset_split,
#     parse_libs,
#     run_d1_inference as run_inference,
# )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eval instruct prompts with an external LLM API")
    parser.add_argument("dataset", help="Local dataset directory or Hugging Face dataset id")
    parser.add_argument("--split", default="v0.1.4")
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-model", default="deepseek-v3.2")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1)  # kept for compatibility
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--filter-third-party", action="store_true")
    parser.add_argument("--python-version", default="3.12")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-retry-missing", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--prompt-mode",
        default="inline",
        choices=["inline", "inline_no_vuln", "requirements.txt", "requirements.txt_no_vuln"],
    )
    parser.add_argument("--api-mode", default="openai", choices=["openai", "openrouter"])
    parser.add_argument("--provider-order", default=None)
    parser.add_argument("--provider-allow-fallbacks", action="store_true")
    parser.add_argument("--http-referer", default=None)
    parser.add_argument("--site-title", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    argv = [
        "evaluate.inference",
        "--track",
        "d1",
        "--dataset",
        args.dataset,
        "--split",
        args.split,
        "--api-model",
        args.api_model,
        "--timeout",
        str(args.timeout),
        "--workers",
        str(args.workers),
        "--python-version",
        args.python_version,
        "--prompt-mode",
        args.prompt_mode,
        "--api-mode",
        args.api_mode,
        "--log-level",
        args.log_level,
        "--sleep",
        str(args.sleep),
    ]
    if args.max_examples is not None:
        argv.extend(["--max-examples", str(args.max_examples)])
    if args.api_url:
        argv.extend(["--api-url", args.api_url])
    if args.api_key:
        argv.extend(["--api-key", args.api_key])
    if args.filter_third_party:
        argv.append("--filter-third-party")
    if args.no_retry_missing:
        argv.append("--no-retry-missing")
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


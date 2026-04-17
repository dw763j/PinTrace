#!/usr/bin/env python
"""Generate JSONL jobs for scripts/batch_pipeline.py.

Features:
- Similar to scripts/gen_pipeline_d2_jobs.py model/provider workflow.
- Works for both tracks: d1 and d2.
- Supports standard and ablation inference modes.
- Can select ablation variants (inline_safe_version / inline_no_vuln / inline_api_rag / all).

Example:
    python -m scripts.gen_batch_pipeline_jobs \
      --output resources/pipeline_d2_ablation_all_py312.jsonl \
      --track d2 \
      --inference-mode ablation \
      --ablation-variant inline_safe_version inline_api_rag \
      --outputs-d2 outputs/d2 \
      --python-version 3.12 \
      --chatbox-key sk-xxx --coding-key sk-xxx --openrouter-key sk-xxx

Then run:
    python -m scripts.batch_pipeline --jobs-jsonl resources/pipeline_d2_ablation_all_py312.jsonl --parallel 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_MODELS_CHATBOX = [
    "Qwen3-235B-A22B-Instruct-2507",
    "Qwen3-30B-A3B-Instruct-2507",
    "Qwen3.5-397B-A17B",
    "gpt-5.4",
    "gemini-3.1-pro-preview",
    "claude-sonnet-4-6",
    "MiniMax-M2.5",
    "DeepSeek-V3.2",
]

DEFAULT_MODELS_KIMI = [
    "kimi-k2.5",
]

DEFAULT_MODELS_OPENROUTER = [
    "meta-llama/llama-4-scout",
]

DEFAULT_MODELS_10 = DEFAULT_MODELS_CHATBOX + DEFAULT_MODELS_KIMI + DEFAULT_MODELS_OPENROUTER

AB_VARIANTS = ["inline_safe_version", "inline_no_vuln", "inline_api_rag"]

CHATBOX_API_URL = "https://chatbox.isrc.ac.cn/api"
CODING_API_URL = "https://coding.dashscope.aliyuncs.com/v1"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate JSONL jobs for scripts.batch_pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--track", choices=["d1", "d2"], default="d2", help="Target track.")
    parser.add_argument(
        "--inference-mode",
        choices=["standard", "ablation"],
        default="standard",
        help="Pass-through to evaluate.pipeline --inference-mode.",
    )

    # Data / output roots
    parser.add_argument("--dataset", default=None, help="Dataset path (mainly for stage-1 standard mode).")
    parser.add_argument("--outputs-d1", default="outputs/d1", help="D1 outputs root.")
    parser.add_argument("--outputs-d2", default="outputs/d2", help="D2 outputs root.")

    # Common pipeline args
    parser.add_argument("--stages", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6], help="Stages list.")
    parser.add_argument("--python-version", default="3.12", help="python_version field.")
    parser.add_argument("--workers", type=int, default=8, help="workers field.")
    parser.add_argument("--timeout", type=int, default=180, help="timeout field.")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional max_examples.")

    # D2 specific controls
    parser.add_argument("--prompt-mode", default="blind", help="D2 prompt_mode (context mode).")
    parser.add_argument(
        "--pinning-modes",
        nargs="+",
        default=["inline", "inline_no_vuln", "requirements.txt"],
        help="D2 pinning_mode list.",
    )

    # D1 standard mode controls
    parser.add_argument(
        "--d1-prompt-modes",
        nargs="+",
        default=["inline", "inline_no_vuln", "requirements.txt"],
        help="D1 prompt_mode list for standard mode.",
    )

    # Ablation controls
    parser.add_argument(
        "--ablation-variant",
        default="inline_safe_version",
        choices=AB_VARIANTS + ["all"],
        help="Single ablation prompt variant (backward compatible), or all.",
    )
    parser.add_argument(
        "--ablation-variants",
        nargs="+",
        default=None,
        choices=AB_VARIANTS + ["all"],
        help="Multiple ablation variants, e.g. --ablation-variants inline_safe_version inline_api_rag.",
    )
    parser.add_argument(
        "--ablation-prompts-jsonl",
        default=None,
        help="Direct ablation prompt JSONL path. If set, overrides variant/root-based path resolution.",
    )
    parser.add_argument(
        "--ablation-prompts-root",
        default="outputs/ablation_prompts",
        help="Ablation prompt root dir; expected structure: <root>/<track>/<variant>.jsonl",
    )

    # Provider + model selection
    parser.add_argument("--chatbox-key", default="", help="API key for chatbox/openai endpoint models.")
    parser.add_argument("--coding-key", default="", help="API key for kimi models.")
    parser.add_argument("--openrouter-key", default="", help="API key for openrouter models.")
    parser.add_argument("--models", nargs="*", default=[], help="Model list. Empty means built-in model set.")

    parser.add_argument("--dry-run", action="store_true", help="Only print planned job count.")
    return parser.parse_args()


def _sanitize_model(model: str) -> str:
    return str(model).replace("/", "_").replace(":", "_")


def _provider_for_model(api_model: str) -> str:
    if api_model in DEFAULT_MODELS_KIMI:
        return "coding"
    if api_model in DEFAULT_MODELS_OPENROUTER:
        return "openrouter"
    return "chatbox"


def _resolve_provider_args(
    *,
    provider: str,
    chatbox_key: str,
    coding_key: str,
    openrouter_key: str,
) -> tuple[str, str, str]:
    if provider == "coding":
        if not coding_key:
            raise ValueError("Kimi models require --coding-key")
        return "openai", CODING_API_URL, coding_key
    if provider == "openrouter":
        if not openrouter_key:
            raise ValueError("OpenRouter models require --openrouter-key")
        return "openrouter", OPENROUTER_API_URL, openrouter_key
    if not chatbox_key:
        raise ValueError("Chatbox models require --chatbox-key")
    return "openai", CHATBOX_API_URL, chatbox_key


def _ablation_variants(args: argparse.Namespace) -> list[str]:
    if args.ablation_variants:
        if "all" in args.ablation_variants:
            return list(AB_VARIANTS)
        # Keep order but de-duplicate.
        deduped: list[str] = []
        seen: set[str] = set()
        for item in args.ablation_variants:
            if item not in seen:
                deduped.append(item)
                seen.add(item)
        return deduped
    if args.ablation_variant == "all":
        return list(AB_VARIANTS)
    return [args.ablation_variant]


def _ablation_jsonl_for(args: argparse.Namespace, variant: str) -> str:
    if args.ablation_prompts_jsonl:
        return args.ablation_prompts_jsonl
    return str(Path(args.ablation_prompts_root) / args.track / f"{variant}.jsonl")


def _build_run_name(track: str, model: str, prompt_mode: str, pinning_mode: str | None = None) -> str:
    model_slug = _sanitize_model(model)
    if track == "d1":
        if prompt_mode != "requirements.txt":
            return f"{model_slug}_{prompt_mode}"
        return f"{model_slug}_req"
    pin_suffix = "requirements_txt" if pinning_mode == "requirements.txt" else (pinning_mode or "inline")
    return f"d2_{model_slug}_{prompt_mode}_{pin_suffix}"


def _build_llm_output_path(
    *,
    track: str,
    outputs_d1: str,
    outputs_d2: str,
    prompt_mode: str,
    pinning_mode: str,
    run_name: str,
    api_model: str,
) -> str:
    model_slug = _sanitize_model(api_model)
    if track == "d1":
        return str(Path(outputs_d1) / prompt_mode / run_name / f"{model_slug}_{prompt_mode}.jsonl")
    return str(Path(outputs_d2) / pinning_mode / run_name / f"{model_slug}_{prompt_mode}.jsonl")


def _base_job(
    *,
    args: argparse.Namespace,
    api_mode: str,
    api_model: str,
    api_url: str,
    api_key: str,
) -> dict:
    return {
        "track": args.track,
        "inference_mode": args.inference_mode,
        "api_mode": api_mode,
        "api_model": api_model,
        "api_url": api_url,
        "api_key": api_key,
        "stages": list(args.stages),
        "python_version": str(args.python_version),
        "workers": int(args.workers),
        "timeout": int(args.timeout),
        "max_examples": args.max_examples,
    }


def _build_jobs_for_model(args: argparse.Namespace, api_mode: str, api_model: str, api_url: str, api_key: str) -> list[dict]:
    jobs: list[dict] = []
    base = _base_job(
        args=args,
        api_mode=api_mode,
        api_model=api_model,
        api_url=api_url,
        api_key=api_key,
    )

    if args.inference_mode == "ablation":
        variants = _ablation_variants(args)
        if args.track == "d1":
            for variant in variants:
                run_name = _build_run_name("d1", api_model, variant)
                job = {
                    **base,
                    "prompt_mode": variant,
                    "outputs_d1": args.outputs_d1,
                    "run_name": run_name,
                    "ablation_prompts_jsonl": _ablation_jsonl_for(args, variant),
                    "llm_output_path": _build_llm_output_path(
                        track="d1",
                        outputs_d1=args.outputs_d1,
                        outputs_d2=args.outputs_d2,
                        prompt_mode=variant,
                        pinning_mode="inline",
                        run_name=run_name,
                        api_model=api_model,
                    ),
                }
                if args.dataset:
                    job["dataset"] = args.dataset
                jobs.append(job)
            return jobs

        for variant in variants:
            # In ablation mode, variant itself is the experimental mode tag.
            # Keep prompt_mode and pinning_mode aligned with the selected variant.
            pin = variant
            run_name = _build_run_name("d2", api_model, variant, pin)
            job = {
                **base,
                "prompt_mode": variant,
                "pinning_mode": pin,
                "outputs_d2": args.outputs_d2,
                "run_name": run_name,
                "ablation_prompts_jsonl": _ablation_jsonl_for(args, variant),
                "llm_output_path": _build_llm_output_path(
                    track="d2",
                    outputs_d1=args.outputs_d1,
                    outputs_d2=args.outputs_d2,
                    prompt_mode=variant,
                    pinning_mode=pin,
                    run_name=run_name,
                    api_model=api_model,
                ),
            }
            if args.dataset:
                job["dataset"] = args.dataset
            jobs.append(job)
        return jobs

    # standard mode
    if args.track == "d1":
        for mode in args.d1_prompt_modes:
            run_name = _build_run_name("d1", api_model, mode)
            job = {
                **base,
                "prompt_mode": mode,
                "outputs_d1": args.outputs_d1,
                "run_name": run_name,
                "llm_output_path": _build_llm_output_path(
                    track="d1",
                    outputs_d1=args.outputs_d1,
                    outputs_d2=args.outputs_d2,
                    prompt_mode=mode,
                    pinning_mode="inline",
                    run_name=run_name,
                    api_model=api_model,
                ),
            }
            if args.dataset:
                job["dataset"] = args.dataset
            jobs.append(job)
        return jobs

    for pin in args.pinning_modes:
        run_name = _build_run_name("d2", api_model, args.prompt_mode, pin)
        job = {
            **base,
            "prompt_mode": args.prompt_mode,
            "pinning_mode": pin,
            "outputs_d2": args.outputs_d2,
            "run_name": run_name,
            "llm_output_path": _build_llm_output_path(
                track="d2",
                outputs_d1=args.outputs_d1,
                outputs_d2=args.outputs_d2,
                prompt_mode=args.prompt_mode,
                pinning_mode=pin,
                run_name=run_name,
                api_model=api_model,
            ),
        }
        if args.dataset:
            job["dataset"] = args.dataset
        jobs.append(job)
    return jobs


def main() -> None:
    args = parse_args()
    models = list(args.models) if args.models else list(DEFAULT_MODELS_10)

    jobs: list[dict] = []
    for model in models:
        provider = _provider_for_model(model)
        api_mode, api_url, api_key = _resolve_provider_args(
            provider=provider,
            chatbox_key=args.chatbox_key,
            coding_key=args.coding_key,
            openrouter_key=args.openrouter_key,
        )
        jobs.extend(_build_jobs_for_model(args, api_mode, model, api_url, api_key))

    if args.dry_run:
        print(f"[dry-run] would generate {len(jobs)} jobs")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False))
            f.write("\n")

    print(f"wrote {len(jobs)} jobs to {args.output}")


if __name__ == "__main__":
    main()

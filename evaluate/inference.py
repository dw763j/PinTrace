"""Unified inference utilities for both D1 and D2 tracks.

Usage:
    python -m evaluate.inference --track d1 ...
    python -m evaluate.inference --track d2 ...
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from loguru import logger
from stdlib_list import stdlib_list

from paths import LOGS, OUTPUTS, ensure_dirs
from stages.openai_api import call_model
from stages.openrouter_api import OPENROUTER_BASE, call_model as call_openrouter
from evaluate.d2.prompt_builder import build_prompt_batch


@dataclass
class LLMConfig:
    api_url: str
    api_key: str = "YOUR_API_KEY"
    api_model: str = "deepseek-v3.2"
    timeout: int = 60
    sleep_seconds: float = 0.0
    api_mode: str = "openai"
    provider: dict | str | None = None
    http_referer: str | None = None

def load_dataset_split(resource: str, split: str) -> Dataset:
    path = Path(resource)
    if path.exists():
        dataset = load_from_disk(str(path))
    else:
        dataset = load_dataset(resource)
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise ValueError(f"Split '{split}' not found. Available: {list(dataset.keys())}")
        return dataset[split]
    if split:
        logger.warning("Split ignored because dataset is not a DatasetDict")
    return dataset


def parse_libs(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        return [str(item) for item in raw if item]
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    if isinstance(value, dict):
        return [str(key) for key in value.keys() if key]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item]
    return []


def filter_third_party(dataset: Dataset, python_version: str) -> Dataset:
    stdlib = set(stdlib_list(python_version))

    def has_third_party(example: dict) -> bool:
        libs = [lib.strip() for lib in parse_libs(example.get("libs")) if lib]
        return any(lib not in stdlib for lib in libs)

    return dataset.filter(has_third_party)


def _extract_third_party_libs(example: dict, stdlib: set[str]) -> list[str]:
    libs = [lib.strip() for lib in parse_libs(example.get("libs")) if lib]
    return sorted({lib for lib in libs if lib not in stdlib})


def augment_prompt_with_versions(base_prompt: str, prompt_mode: str = "inline") -> str:
    index = base_prompt.find("You should write self-contained code starting with:")
    if prompt_mode == "inline":
        suffix = (
            "Ensure every import statement for the utilized third-party libraries (except python stdlib) "
            "includes a trailing comment of the form '# VERSION=x.y.z' with the version you select. "
            "Attach the comment directly to the import line (e.g., 'import xxyyzz  # VERSION=x.y.z'). "
            "GENERATE CODE DIRECTLY WITHOUT ANY EXPLANATION.\n"
        )
    elif prompt_mode == "inline_no_vuln":
        suffix = (
            "Ensure every import statement for the utilized third-party libraries (except python stdlib) "
            "includes a trailing comment of the form '# VERSION=x.y.z' with the version you select. "
            "Attach the comment directly to the import line (e.g., 'import xxyyzz  # VERSION=x.y.z'), "
            "do not use vulnerable version. GENERATE CODE DIRECTLY WITHOUT ANY EXPLANATION.\n"
        )
    elif prompt_mode == "requirements.txt":
        suffix = (
            "Provide a requirements.txt file that lists all the third-party libraries you use to solve "
            "the task within a block of ```requirements.txt ...```. GENERATE CODE DIRECTLY WITHOUT "
            "ANY EXPLANATION.\n"
        )
    elif prompt_mode == "requirements.txt_no_vuln":
        suffix = (
            "Provide a requirements.txt file that lists all the third-party libraries you use to solve "
            "the task within a block of ```requirements.txt ...```. Do not use vulnerable version. "
            "GENERATE CODE DIRECTLY WITHOUT ANY EXPLANATION.\n"
        )
    else:
        raise ValueError(f"Invalid prompt mode: {prompt_mode}")
    if index != -1:
        return f"{base_prompt[:index]}{suffix}{base_prompt[index:]}"
    return f"{base_prompt}{suffix}"


def _call_llm(
    *,
    prompt: str,
    config: LLMConfig,
) -> dict:
    if config.api_mode == "openrouter":
        return call_openrouter(
            token=config.api_key,
            api_address=config.api_url or OPENROUTER_BASE,
            model=config.api_model,
            message=prompt,
            timeout=float(config.timeout),
            provider=config.provider,
            http_referer=config.http_referer
        )
    return call_model(
        token=config.api_key,
        api_address=config.api_url,
        model=config.api_model,
        message=prompt,
        timeout=float(config.timeout),
    )


def _is_d1_record_valid(record: dict) -> bool:
    out = record.get("llm_output")
    if not isinstance(out, dict):
        return False
    content = out.get("content")
    return isinstance(content, str) and bool(content.strip())


def _load_existing_d1_valid_records(output_path: Path) -> tuple[dict[str, dict], set[str]]:
    existing: dict[str, dict] = {}
    valid_ids: set[str] = set()
    if not output_path.exists():
        return existing, valid_ids
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = rec.get("task_id")
            if isinstance(tid, str):
                existing[tid] = rec
                if _is_d1_record_valid(rec):
                    valid_ids.add(tid)
    return existing, valid_ids


def run_d1_inference(
    *,
    dataset: Dataset,
    config: LLMConfig,
    output_path: Path,
    max_examples: Optional[int],
    prompt_mode: str,
    python_version: str,
    max_workers: int = 4,
    retry_missing: bool = True,
    max_retries: int = 3,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stdlib = set(stdlib_list(python_version))
    dataset_length = len(dataset)
    items_all = list(dataset)
    if max_examples is not None:
        items_all = items_all[:max_examples]
    expected_ids = {item["task_id"] for item in items_all}
    existing_records, valid_ids = _load_existing_d1_valid_records(output_path)
    missing_ids = expected_ids - valid_ids
    items_to_process = [item for item in items_all if item["task_id"] in missing_ids]
    if not items_to_process:
        logger.info(f"All {len(expected_ids)} tasks already present with valid output.")
        return
    if retry_missing and missing_ids:
        logger.info(f"Resuming D1 inference: done={len(valid_ids)} missing={len(missing_ids)}")

    def _process(item: dict) -> dict:
        prompt = item["instruct_prompt"]
        augmented_prompt = augment_prompt_with_versions(prompt, prompt_mode)
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                result = _call_llm(prompt=augmented_prompt, config=config)
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < max_retries - 1:
                    delay = 2.0 * (attempt + 1)
                    logger.warning(
                        f"Task {item['task_id']} attempt {attempt + 1}/{max_retries} failed: {exc}; retry in {delay}s"
                    )
                    time.sleep(delay)
                else:
                    raise last_err
        rec = {
            "llm_config": {
                "api_url": config.api_url,
                "api_model": config.api_model,
                "python_version": python_version,
                "dataset_length": dataset_length,
            },
            "task_id": item["task_id"],
            "instruct_prompt": prompt,
            "augmented_prompt": augmented_prompt,
            "third_party_libs": _extract_third_party_libs(item, stdlib),
            "llm_output": result,
        }
        if config.sleep_seconds:
            time.sleep(config.sleep_seconds)
        return rec

    write_lock = threading.Lock()
    mode = "a" if output_path.exists() and existing_records else "w"
    written = 0
    with output_path.open(mode, encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process, item): item["task_id"] for item in items_to_process}
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    record = future.result()
                    with write_lock:
                        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                        fh.flush()
                        written += 1
                        logger.info(f"[{written}/{len(items_to_process)}] wrote task_id={task_id}")
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"Failed task_id={task_id}: {exc}")


def load_d2_records_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_written_d2_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    ids: set[str] = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = obj.get("record_id")
            if rid:
                ids.add(rid)
    return ids


def run_d2_inference(
    *,
    records: list[dict],
    output_path: Path,
    config: LLMConfig,
    mode: str,
    pinning_mode: str,
    max_hint_chars: int,
    workers: int,
    max_examples: int | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_items = build_prompt_batch(records, mode=mode, pinning_mode=pinning_mode, max_hint_chars=max_hint_chars)
    for item in prompt_items:
        item["prompt_mode"] = mode
        item["pinning_mode"] = pinning_mode
    written_ids = _load_written_d2_ids(output_path)
    todo = [item for item in prompt_items if item["record_id"] not in written_ids]
    if max_examples is not None:
        todo = todo[:max_examples]
    logger.info(
        f"D2 inference: total={len(prompt_items)} skip={len(written_ids)} todo={len(todo)} "
        f"context={mode} pinning={pinning_mode} model={config.api_model} workers={workers}"
    )
    if not todo:
        return

    write_lock = threading.Lock()
    written = 0
    failed = 0

    def _call_item(item: dict) -> dict:
        result = _call_llm(prompt=item["prompt"], config=config)
        return {
            **item,
            "llm_output": result.get("content", ""),
            "usage": result.get("usage", {}),
            "response_time": result.get("response_time", 0.0),
            "llm_config": {
                "api_url": config.api_url,
                "api_model": config.api_model,
            },
        }

    with output_path.open("a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_call_item, item): item["record_id"] for item in todo}
            for future in as_completed(futures):
                rid = futures[future]
                try:
                    record = future.result()
                    with write_lock:
                        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                        fh.flush()
                        written += 1
                        logger.info(f"[{written}/{len(todo)}] wrote record_id={rid}")
                except Exception as exc:  # noqa: BLE001
                    with write_lock:
                        failed += 1
                    logger.error(f"Failed record_id={rid}: {exc}")
    logger.success(f"D2 inference finished: written={written}, failed={failed}, output={output_path}")


def _build_provider_config(provider_order: str | None, allow_fallbacks: bool) -> dict | None:
    if not provider_order:
        return None
    return {
        "order": [p.strip() for p in provider_order.split(",") if p.strip()],
        "allow_fallbacks": allow_fallbacks,
    }


def _resolve_api_url(api_url: str | None, api_mode: str) -> str:
    if api_url:
        return api_url
    if api_mode == "openrouter":
        return OPENROUTER_BASE
    raise ValueError("--api-url is required for non-openrouter mode")


def _resolve_api_key(arg_key: str | None) -> str:
    return (
        arg_key
        or os.getenv("OPENROUTER_API_KEY", "")
        or os.getenv("API_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
        or os.getenv("CHATBOX_TOKEN", "")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified inference entry for D1/D2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--track", choices=["d1", "d2"], required=True)
    parser.add_argument("--dataset", required=True, help="D1 resource id/path or D2 jsonl path.")
    parser.add_argument("--split", default="v0.1.4", help="D1 split; ignored by D2.")
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-model", default="deepseek-v3.2")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--api-mode", default="openai", choices=["openai", "openwebui", "openrouter"])
    parser.add_argument("--provider-order", default=None)
    parser.add_argument("--provider-allow-fallbacks", action="store_true")
    parser.add_argument("--http-referer", default=None)
    parser.add_argument("--site-title", default=None)
    parser.add_argument("--python-version", default="3.12")
    parser.add_argument("--output-path", default=None, help="If set, write jsonl exactly here.")
    parser.add_argument("--output-dir", default=None, help="Root output directory when output-path unset.")
    parser.add_argument("--filter-third-party", action="store_true")
    parser.add_argument("--prompt-mode", default=None, help="D1 prompt mode or D2 context mode.")
    parser.add_argument("--pinning-mode", default="inline", help="D2 pinning mode.")
    parser.add_argument("--max-hint-chars", type=int, default=1200)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--no-retry-missing", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())
    api_key = _resolve_api_key(args.api_key)
    if not api_key:
        logger.warning("No API key found; requests may fail with 401.")
    api_url = _resolve_api_url(args.api_url, args.api_mode)
    provider = _build_provider_config(args.provider_order, args.provider_allow_fallbacks)
    config = LLMConfig(
        api_url=api_url,
        api_key=api_key,
        api_model=args.api_model,
        timeout=args.timeout,
        sleep_seconds=args.sleep,
        api_mode=args.api_mode,
        provider=provider,
        http_referer=args.http_referer
    )

    if args.track == "d1":
        prompt_mode = args.prompt_mode or "inline"
        dataset = load_dataset_split(args.dataset, args.split)
        if args.filter_third_party:
            dataset = filter_third_party(dataset, args.python_version)
        model_slug = args.api_model.replace("/", "_").replace(":", "_")
        output_path = (
            Path(args.output_path)
            if args.output_path
            else (Path(args.output_dir) if args.output_dir else OUTPUTS / "test_run_inference_d1") / f"{model_slug}.jsonl"
        )
        logger.add(str(LOGS / f"bigcodebench_eval_{model_slug}.log"), level=args.log_level.upper())
        run_d1_inference(
            dataset=dataset,
            config=config,
            output_path=output_path,
            max_examples=args.max_examples,
            prompt_mode=prompt_mode,
            python_version=args.python_version,
            max_workers=args.workers,
            retry_missing=not args.no_retry_missing,
        )
        logger.success(f"D1 inference finished: {output_path}")
        return

    mode = args.prompt_mode or "blind"
    records = load_d2_records_jsonl(args.dataset)
    model_slug = args.api_model.replace("/", "_").replace(":", "_")
    output_path = (
        Path(args.output_path)
        if args.output_path
        else ((Path(args.output_dir) if args.output_dir else OUTPUTS / "d2") / args.pinning_mode / f"{model_slug}_{mode}.jsonl")
    )
    logger.add(
        str(LOGS / f"d2_inference_{model_slug}_{args.pinning_mode}_{mode}.log"),
        level=args.log_level.upper(),
    )
    run_d2_inference(
        records=records,
        output_path=output_path,
        config=config,
        mode=mode,
        pinning_mode=args.pinning_mode,
        max_hint_chars=args.max_hint_chars,
        workers=args.workers,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()


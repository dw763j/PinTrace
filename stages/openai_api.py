"""OpenAI-compatible API via the official openai library. Implements llm_api interface.

Usage (import-only)::

    from stages.openai_api import call_model
"""

from __future__ import annotations

import json
import time
from typing import Any, NoReturn

from loguru import logger
from openai import OpenAI

# Exception types (openai >=1.0)
try:
    from openai import APIError, APIConnectionError, APITimeoutError, RateLimitError
except (ImportError, AttributeError):
    APIError = APIConnectionError = APITimeoutError = RateLimitError = None # type: ignore


def _wrap_api_error(e: Exception) -> NoReturn:
    """Raise a wrapped exception with clearer messages. Preserves cause chain."""
    msg = str(e)
    if RateLimitError is not None and isinstance(e, RateLimitError):
        raise Exception(f"Rate limit exceeded: {msg}") from e
    if APITimeoutError is not None and isinstance(e, APITimeoutError):
        raise Exception(f"Request timeout: {msg}") from e
    if APIConnectionError is not None and isinstance(e, APIConnectionError):
        raise Exception(f"Connection failed: {msg}") from e
    if APIError is not None and isinstance(e, APIError):
        raise Exception(f"API error: {msg}") from e
    raise Exception(f"Request failed (retries exhausted): {msg}") from e


def call_model(
    token: str,
    api_address: str,
    model: str = "gpt-4o-mini",
    message: str = "Why can't I see the light?",
    max_retries: int = 5,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Call chat completion API via openai library. Same interface as llm_api.call_model."""
    client = OpenAI(
        api_key=token or "sk-placeholder",
        base_url=api_address,
        max_retries=max_retries,
    )
    start = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message}],
            timeout=timeout,
        )
    except Exception as e:
        _wrap_api_error(e)

    elapsed = round(time.perf_counter() - start, 2)
    choice = (resp.choices or [None])[0]
    if not choice or not getattr(choice, "message", None):
        raise Exception(f"Invalid response: missing choices, id={getattr(resp, 'id', '?')}")
    content = (choice.message.content or "") if choice.message else ""
    usage = {}
    if resp.usage:
        usage = resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else {"prompt_tokens": getattr(resp.usage, "prompt_tokens", 0), "completion_tokens": getattr(resp.usage, "completion_tokens", 0)}
    return {"content": content, "usage": usage, "response_time": elapsed}


def get_models(token: str, api_address: str) -> list | None:
    """List available models. For openai-compatible APIs."""
    try:
        client = OpenAI(api_key=token or "sk-placeholder", base_url=api_address)
        models = client.models.list()
        if not models.data:
            return []
        return [m.model_dump() if hasattr(m, "model_dump") else {"id": getattr(m, "id", str(m))} for m in models.data]
    except Exception as e:
        logger.warning(f"get_models failed: {e}")
        return None


def llm_output_to_json(output: str | None) -> dict[str, Any] | None:
    """Parse LLM output string to JSON dict."""
    if output is None or not isinstance(output, str):
        return None
    s = output.strip().replace("\n", "")
    if not s:
        return None
    if s.startswith("```json"):
        s = s[7:].strip()
    if s.endswith("```"):
        s = s[:-3].strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        logger.debug(f"llm_output_to_json parse failed: {e}")
        return None

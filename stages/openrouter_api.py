"""OpenRouter API client. Same interface as openai_api.call_model with extra provider support.

Usage (import-only)::

    from stages.openrouter_api import call_model
"""

from __future__ import annotations

import time
from typing import Any

import requests
from loguru import logger

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def call_model(
    token: str,
    api_address: str = OPENROUTER_BASE,
    model: str = "mistralai/mixtral-8x7b-instruct",
    message: str = "Hello",
    max_retries: int = 5,
    timeout: float = 120.0,
    *,
    provider: dict[str, Any] | str | None = None,
    http_referer: str | None = None,
    site_title: str | None = None,
) -> dict[str, Any]:
    """Call OpenRouter chat completion API.

    Args:
        token: OpenRouter API key (Bearer token)
        api_address: Base URL, default https://openrouter.ai/api/v1
        model: Model id (e.g. mistralai/mixtral-8x7b-instruct)
        message: User message content
        max_retries: Max retries on failure
        provider: Provider name (str, e.g. "groq") or full config dict, e.g. {"order": ["openai", "together"], "allow_fallbacks": False}
        http_referer: Optional HTTP-Referer header
        site_title: Optional X-OpenRouter-Title header

    Returns:
        {"content": str, "usage": dict, "response_time": float}
    """
    url = f"{api_address.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {token or ''}",
        "Content-Type": "application/json",
    }
    if http_referer:
        headers["HTTP-Referer"] = http_referer
    if site_title:
        headers["X-OpenRouter-Title"] = site_title

    data: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
    }
    if provider is not None:
        # provider may be str (single provider) or dict (full config)
        if isinstance(provider, str):
            provider_out = {"order": [provider.strip()], "allow_fallbacks": False}
        elif isinstance(provider, dict):
            provider_out = dict(provider)
            if "allow_fallbacks" not in provider_out:
                provider_out["allow_fallbacks"] = False
        else:
            provider_out = {"order": [str(provider)], "allow_fallbacks": False}
        data["provider"] = provider_out

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            start = time.perf_counter()
            resp = requests.post(url, headers=headers, json=data, timeout=timeout)
            elapsed = round(time.perf_counter() - start, 2)
        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(1.0 * (attempt + 1))
            continue

        if resp.status_code != 200:
            msg = f"OpenRouter error {resp.status_code}: {resp.text}"
            try:
                err_body = resp.json()
                if "error" in err_body and isinstance(err_body["error"], dict):
                    err_detail = err_body["error"].get("message", resp.text)
                    msg = f"OpenRouter error {resp.status_code}: {err_detail}"
            except Exception:
                pass
            raise Exception(msg)

        body = resp.json()
        choices = body.get("choices", [])
        if not choices:
            raise Exception(f"Invalid response: missing choices, body={body}")

        choice = choices[0] or {}
        msg_obj = choice.get("message") or {}
        content = msg_obj.get("content") or ""
        usage = body.get("usage") or {}

        return {"content": content, "usage": usage, "response_time": elapsed}

    raise Exception(f"Request failed after {max_retries} retries: {last_err}") from last_err


def get_models(token: str, api_address: str = OPENROUTER_BASE) -> list[dict[str, Any]] | None:
    """List models available on OpenRouter."""
    url = f"{api_address.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {token or ''}"}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            logger.warning(f"OpenRouter models list failed: {resp.status_code}")
            return None
        body = resp.json()
        data = body.get("data") or []
        return data if isinstance(data, list) else None
    except Exception as e:
        logger.warning(f"get_models failed: {e}")
        return None

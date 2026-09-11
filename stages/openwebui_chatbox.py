import requests
import os
import time
import json
from typing import Any
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _build_retry_session(max_retries: int, base_delay: float) -> requests.Session:
    """Create a requests session with urllib3 retry policy."""
    retry = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        status=max_retries,
        backoff_factor=base_delay,
        status_forcelist=(408, 409, 425, 429, 500, 502, 503, 504),
        allowed_methods=frozenset({"POST", "GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    # Keep deterministic network behavior in this project context.
    session.trust_env = False
    return session

def call_model(
    token: str,
    api_address: str,
    model: str = "Qwen/Qwen2.5-14B-Instruct",
    message: str = "Why is the sky blue?",
    max_retries: int = 5,
    base_delay: float = 1.0,
    api_mode: str = "openwebui",
) -> dict[str, Any]:
    """Call a chat-completions API with retries.

    Args:
        token: API token.
        api_address: API base URL.
        model: Model name.
        message: User prompt.
        max_retries: Maximum retry attempts.
        base_delay: Base delay in seconds between retries.

    Returns:
        A dict containing the model response.

    Raises:
        Exception: If every retry fails.
    """
    base = api_address.rstrip("/")
    if api_mode == "openai":
        url = f"{base}/chat/completions"
    elif api_mode == "openwebui":
        url = f"{base}/api/chat/completions"
    else:
        raise ValueError(f"Unsupported api_mode: {api_mode}")

    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": message,
            }
        ]
    }

    try:
        with _build_retry_session(max_retries=max_retries, base_delay=base_delay) as session:
            start_time = time.time()
            response = session.post(url, headers=headers, json=data, timeout=120)
            response_time = time.time() - start_time
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed after retries: {e}") from e

    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code} - {response.text}")

    body = response.json()
    choices = body.get("choices", [])
    if not choices:
        raise Exception(f"Invalid response payload: missing choices, body={body}")
    content = ((choices[0] or {}).get("message") or {}).get("content")
    usage = body.get("usage", {})
    return {
        "content": content,
        "usage": usage,
        "response_time": round(response_time, 2),
    }


def get_models(token: str, api_address: str) -> list | None:
    """List available models.

    Args:
        token: API token.
        api_address: API base URL.

    Returns:
        Model list, or None if the request fails.
    """
    try:
        response = requests.get(
            f'{api_address}/api/models',
            headers={
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            }
        )
        # Check HTTP status.
        if response.status_code != 200:
            logger.error(f"Error: {response.status_code} - {response.text}")
            return None
            
        # Parse JSON.
        data = response.json()['data']
        # for item in data:
        #     # Drop profile_image_url if present.
        #     if 'profile_image_url' in item['info'].get('meta', {}):
        #         del item['info']['meta']['profile_image_url']
        #     # Convert created_at and updated_at.
        #     for key in ['created_at', 'updated_at']:
        #         ts = item['info'].get(key)
        #         if isinstance(ts, int | float):
        #             item['info'][key] = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        #         elif isinstance(ts, str):
        #             # Already a string; skip.
        #             pass

        # Write a fresh JSON file.
        id_list = []
        for item in data:
            id_list.append(item['id'])
        data.append({'id_list': sorted(id_list)})
        with open('chatbox_models.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        return data
        
    except Exception as e:
        logger.error(f"Error while listing models: {str(e)}")
        return None

def llm_output_to_json(output: str) -> dict[str, Any]:
    """Parse LLM output as JSON."""
    output = output.strip().replace('\n', '')
    if output.startswith('```json'):
        output = output[7:].strip()
    if output.endswith('```'):
        output = output[:-3].strip()
    try:
        json_output = json.loads(output)
        return json_output
    except Exception as e:
        logger.error(f"Error converting LLM output to JSON: {str(e)}")
        return {}

if __name__ == "__main__":
    token = os.getenv("CHATBOX_TOKEN")
    if not token:
        raise ValueError("Set the CHATBOX_TOKEN environment variable")
    get_models(token, "https://chatbox.isrc.ac.cn")


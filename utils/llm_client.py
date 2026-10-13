"""
LLM Client — centralised OpenAI API wrapper with retry logic.
"""
from __future__ import annotations

import base64
import time
from typing import Optional

import openai

import config


_client: Optional[openai.OpenAI] = None


def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def chat_completion(
    system_prompt: str,
    user_message: str,
    model: str = config.OPENAI_MODEL,
    temperature: float = config.OPENAI_TEMPERATURE,
    max_retries: int = 3,
    json_mode: bool = False,
) -> str:
    """
    Send a chat completion request and return the assistant's text.

    Args:
        system_prompt: The system-level instruction.
        user_message:  The user turn content.
        model:         OpenAI model name.
        temperature:   Sampling temperature.
        max_retries:   Number of retries on transient errors.
        json_mode:     If True, request JSON response format.

    Returns:
        The assistant response as a plain string.
    """
    client = _get_client()

    kwargs: dict = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_message},
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except openai.RateLimitError:
            wait = 2 ** attempt
            time.sleep(wait)
        except openai.APIError as e:
            if attempt == max_retries:
                raise
            time.sleep(1)

    raise RuntimeError("LLM call failed after maximum retries.")


def vision_page_transcription(
    system_prompt: str,
    image_png_bytes: bytes,
    *,
    user_instruction: str = "Transcribe all visible educational text from this page.",
    model: Optional[str] = None,
    detail: str = "high",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> str:
    """
    Send one page image (PNG bytes) to a vision-capable chat model (e.g. gpt-4o-mini).
    """
    model_name = model or getattr(config, "PDF_VISION_MODEL", None) or config.OPENAI_MODEL
    b64 = base64.standard_b64encode(image_png_bytes).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    client = _get_client()
    kwargs = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": detail},
                    },
                    {"type": "text", "text": user_instruction},
                ],
            },
        ],
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            return (content or "").strip()
        except openai.RateLimitError:
            time.sleep(2**attempt)
        except openai.APIError:
            if attempt == max_retries:
                raise
            time.sleep(1)

    raise RuntimeError("Vision transcription failed after maximum retries.")

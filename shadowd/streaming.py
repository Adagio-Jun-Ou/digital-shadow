#!/usr/bin/env python3
"""
SSE Streaming support for Shadow AI LLM responses.

Implements Server-Sent Events endpoint that streams LLM tokens
as they arrive, following the OpenAI-compatible SSE protocol.
"""

import asyncio
import json
import re
import time
import logging
from typing import AsyncIterator, Optional

logger = logging.getLogger("shadowd.streaming")

_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _sanitize_content(text: str, maxlen: int = 10000) -> str:
    if not isinstance(text, str):
        return ""
    return _SANITIZE_CONTENT_RE.sub('', text)[:maxlen].strip()


def _mask_key(key: str) -> str:
    """Mask an API key for logging. Shows first 4 and last 4 chars, rest masked."""
    if not key or len(key) <= 8:
        return "***"
    return key[:4] + "..." + key[-4:]


async def generate_sse_stream(
    content_generator: AsyncIterator[str],
    request_id: Optional[str] = None,
    model: str = "glm-5",
) -> AsyncIterator[str]:
    """
    Convert an async token generator into SSE-formatted lines.
    
    Follows the OpenAI chat completion streaming format:
    data: {"id":"...","object":"chat.completion.chunk","created":...,"model":"...",
           "choices":[{"index":0,"delta":{"content":"token"},"finish_reason":null}]}
    data: [DONE]
    """
    created = int(time.time())
    chunk_id = _sanitize_content(request_id, 200) if request_id else f"chatcmpl-{created}"
    model = _sanitize_content(model, 100)
    
    try:
        async for token in content_generator:
            chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        
        final_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        
    except asyncio.CancelledError:
        logger.info("SSE stream cancelled for %s", chunk_id)
        error_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "cancelled"
            }]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error("SSE stream error: %s", type(e).__name__)
        error_chunk = {
            "id": chunk_id,
            "object": "error",
            "created": created,
            "error": {"message": "Stream error occurred", "type": "stream_error"}
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"


async def stream_llm_response(client, messages: list, model: str = "glm-5",
                              temperature: float = 0.7, max_tokens: int = 2048,
                              api_base: str = "", api_key: str = "") -> AsyncIterator[str]:
    """
    Stream LLM response token by token.
    
    Uses httpx streaming to get tokens as they arrive from the API.
    """
    temperature = max(0.0, min(2.0, temperature))
    max_tokens = max(1, min(32768, max_tokens))
    try:
        async with client.stream(
            "POST",
            f"{api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True
            },
            timeout=60.0
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data)
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        logger.error("LLM stream error for %s: %s", _mask_key(api_key), type(e).__name__)
        yield "\n[Stream error occurred]"
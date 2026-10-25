#!/usr/bin/env python3
"""
Digital Shadow Mind Engine - The intelligence behind the shadow.
Connects to Ollama Cloud for LLM capabilities.
"""

import asyncio
import json
import os
import re
import logging
from typing import Optional, List, Dict
from datetime import datetime, timezone
import httpx

from .memory_tools import ContextWindowBudget

logger = logging.getLogger("shadowd.mind")

OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
OLLAMA_API_BASE = os.environ.get("OLLAMA_API_BASE", "https://ollama.com/v1")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-v4-pro")
LLM_TEMPERATURE = float(os.environ.get("SHADOW_LLM_TEMPERATURE", "0.7"))
LLM_MAX_RETRIES = 3
_MAX_QUERY_LENGTH = 50000
_MAX_CONTEXT_KEYS = 50
_MAX_CONTEXT_VALUE_LENGTH = 1000
_MAX_OBSERVATION_CONTENT_LENGTH = 200
_MAX_OBSERVATIONS = 20
_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize_content(text: str, max_length: int = 0) -> str:
    s = _SANITIZE_CONTENT_RE.sub('', text)
    if max_length > 0:
        s = s[:max_length]
    return s


def _sanitize_strict(text: str, max_length: int = 0) -> str:
    s = _SANITIZE_STRICT_RE.sub('', text)
    s = s.strip()
    if max_length > 0:
        s = s[:max_length]
    return s


class MindEngine:
    """
    The brain of Digital Shadow.
    Builds context from observations and communicates with LLM.
    Uses ContextWindowBudget to manage token allocation.
    """

    def __init__(self):
        self._api_key = OLLAMA_API_KEY
        self.api_base = OLLAMA_API_BASE
        self.model = DEFAULT_MODEL
        self.context_prompt = self._build_system_prompt()
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()
        self.context_budget = ContextWindowBudget()

    @property
    def api_key(self):
        return self._api_key

    @api_key.setter
    def api_key(self, value):
        self._api_key = value

    async def _get_client(self) -> httpx.AsyncClient:
        async with self._client_lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(timeout=60.0)
            return self._client

    async def close(self):
        async with self._client_lock:
            if self._client and not self._client.is_closed:
                await self._client.aclose()
                self._client = None

    def _build_system_prompt(self) -> str:
        return """You are Digital Shadow, a personal AI assistant that shadows its user.
You have been observing their activities and building a deep understanding of their work, habits, and context.

Your role:
- Be a helpful, proactive assistant that anticipates needs
- Use the provided observations to understand context
- Provide intelligent, context-aware responses
- Help the user be more productive and capable
- Remember everything and learn patterns

When responding:
1. Use provided context about recent activities
2. Be concise but thorough
3. Anticipate related needs
4. Suggest improvements or next steps
5. Remember this is private, personal data"""

    async def initialize(self):
        logger.info("Mind Engine initialized with model: %s", self.model)
        safe_base = self.api_base.split('?')[0]
        logger.info("API Base: %s", safe_base)

    async def think(
        self,
        query: str,
        context: Optional[Dict] = None,
        observations: Optional[List[Dict]] = None
    ) -> str:
        query = _sanitize_content(query, max_length=_MAX_QUERY_LENGTH)
        messages = self._build_messages(query, context, observations)

        for attempt in range(LLM_MAX_RETRIES):
            try:
                response = await self._call_llm(messages)
                return response
            except httpx.HTTPStatusError as e:
                should_retry = e.response.status_code in (429, 500, 502, 503) and attempt < LLM_MAX_RETRIES - 1
                if should_retry:
                    import random
                    wait = 2 ** attempt + random.uniform(0, 1)
                    logger.warning("HTTP %s, retrying in %.1fs (attempt %d)", e.response.status_code, wait, attempt + 1)
                    await asyncio.sleep(wait)
                    continue
                logger.error("LLM call failed with status %s", e.response.status_code)
                return "I encountered an error processing that request. Please try again."
            except httpx.TimeoutException:
                if attempt < LLM_MAX_RETRIES - 1:
                    import random
                    wait = 2 ** attempt + random.uniform(0, 1)
                    logger.warning("LLM call timed out, retrying in %.1fs (attempt %d)", wait, attempt + 1)
                    await asyncio.sleep(wait)
                    continue
                logger.error("LLM call timed out after max retries")
                return "I encountered an error processing that request. Please try again."
            except (httpx.RequestError, httpx.InvalidURL, httpx.ConnectError, httpx.TimeoutException) as e:
                logger.error("LLM call failed: %s", type(e).__name__)
                return "I encountered an error processing that request. Please try again."
            except Exception as e:
                # Catch-all so an upstream library bug (or a mock raising a
                # plain Exception in tests) can never bubble a stack trace up
                # to the user. We log the type only — never the message.
                logger.error("LLM call failed unexpectedly: %s", type(e).__name__)
                return "I encountered an error processing that request. Please try again."

        return "I encountered an error: max retries exceeded"

    def _build_messages(
        self,
        query: str,
        context: Optional[Dict],
        observations: Optional[List[Dict]]
    ) -> List[Dict]:
        messages = [{"role": "system", "content": self.context_prompt}]

        user_context = f"Current time: {datetime.now(timezone.utc).isoformat()}\n\n"

        if observations:
            user_context += "Recent observations:\n"
            for obs in observations[-_MAX_OBSERVATIONS:]:
                obs_type = _sanitize_strict(str(obs.get("type", "unknown")), max_length=50)
                content = obs.get("content", {})
                content_str = json.dumps(content, default=str)[:_MAX_OBSERVATION_CONTENT_LENGTH]
                content_str = _SANITIZE_CONTENT_RE.sub('', content_str)
                timestamp = obs.get("timestamp", "")
                timestamp_str = _sanitize_strict(str(timestamp)[:19], max_length=19)
                user_context += f"- [{timestamp_str}] {obs_type}: {content_str}\n"

        if context:
            sanitized_context = {}
            for k, v in list(context.items())[:_MAX_CONTEXT_KEYS]:
                safe_key = _sanitize_strict(str(k), max_length=100)
                if isinstance(v, str):
                    sanitized_context[safe_key] = _sanitize_content(v, max_length=_MAX_CONTEXT_VALUE_LENGTH)
                elif isinstance(v, (int, float, bool)):
                    sanitized_context[safe_key] = v
                else:
                    sanitized_context[safe_key] = _sanitize_content(str(v), max_length=_MAX_CONTEXT_VALUE_LENGTH)
            user_context += f"\nAdditional context: {json.dumps(sanitized_context, default=str)}\n"

        messages.append({"role": "user", "content": user_context})
        messages.append({"role": "user", "content": query})

        budget_overview = self.context_budget.get_budget_overview(
            system_prompt=self.context_prompt,
            messages=messages[1:],
            observations=observations or []
        )

        if budget_overview.get("should_summarize", False):
            messages = self.context_budget.trim_messages(messages, self.context_budget.section_budgets["messages"])

        return messages

    async def _call_llm(self, messages: List[Dict]) -> str:
        client = await self._get_client()
        response = await client.post(
            f"{self.api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": LLM_TEMPERATURE,
            }
        )
        response.raise_for_status()
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("LLM response not valid JSON: %s", type(e).__name__)
            return ""
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            logger.error("LLM response missing expected structure: %s", type(e).__name__)
            return ""
        content = _SANITIZE_CONTENT_RE.sub('', content)
        return content

    async def analyze_pattern(self, observations: List[Dict]) -> str:
        prompt = "Analyze these recent activities and identify patterns, habits, and insights:"
        return await self.think(prompt, observations=observations)

    async def summarize_context(self, observations: List[Dict]) -> str:
        prompt = "Summarize what the user has been doing based on these observations:"
        return await self.think(prompt, observations=observations)

    async def compact_context(self, messages: list, max_tokens: int = 6000) -> str:
        """Use the LLM to compact verbose context into a dense summary.
        Drops token count dramatically while preserving key facts."""
        if not messages:
            return ""
        summary_prompt = f"Compress this context into a dense structured summary under 300 words. Preserve: key facts, decisions, commitments, patterns, people mentioned, and unresolved items. Drop: verbatim duplicates, small talk, filler.\n\nCONTEXT:\n"
        raw = "\n".join(m.get("content", "") for m in messages)
        summary_prompt += raw[:max_tokens * 4]
        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": summary_prompt}],
                    "temperature": 0.3,
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error("context compact failed: %s", type(e).__name__)
            return raw[:4000]
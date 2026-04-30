"""
generation/llm_client.py

LLM API wrapper for the FinGuard generation layer.

Primary:  Groq  (Llama-3 70B or Mixtral) — fast inference, free tier
Fallback: OpenAI (GPT-4o-mini) — higher quality, paid

Features:
  - Config-driven model and provider selection
  - Exponential backoff retry on transient errors (rate limit, timeout, 5xx)
  - Hard 10-second timeout per attempt — never blocks the pipeline
  - Automatic provider fallback: Groq fails N times → try OpenAI
  - Returns a structured LLMResponse so callers never need to inspect exceptions

Environment variables required:
  GROQ_API_KEY   — get free at https://console.groq.com
  OPENAI_API_KEY — optional, only needed if fallback is enabled
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"

HARD_TIMEOUT_SECONDS = 10.0
MAX_RETRIES_PRIMARY = 3
MAX_RETRIES_FALLBACK = 2
BASE_BACKOFF_SECONDS = 0.5


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    retries_used: int
    error: Optional[str] = None
    success: bool = True


def _exponential_backoff(attempt: int, base: float = BASE_BACKOFF_SECONDS) -> float:
    return min(base * (2 ** attempt), 8.0)


def _call_groq(
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> dict:
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("pip install groq")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")

    client = Groq(api_key=api_key, timeout=timeout)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return {
        "text": response.choices[0].message.content.strip(),
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }


def _call_openai(
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> dict:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key, timeout=timeout)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return {
        "text": response.choices[0].message.content.strip(),
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }


_PROVIDER_CALLERS = {
    "groq": _call_groq,
    "openai": _call_openai,
}

_RETRYABLE_EXCEPTIONS = (
    "rate_limit",
    "timeout",
    "connection",
    "server_error",
    "service_unavailable",
)


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in _RETRYABLE_EXCEPTIONS) or isinstance(
        exc, (TimeoutError, ConnectionError)
    )


def _call_with_retry(
    caller,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    max_retries: int,
    provider_name: str,
) -> tuple[Optional[dict], int, Optional[str]]:
    last_error = None
    retries = 0

    for attempt in range(max_retries + 1):
        try:
            result = caller(prompt, model, max_tokens, temperature, timeout)
            return result, retries, None
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries and _is_retryable(exc):
                wait = _exponential_backoff(attempt)
                log.warning(
                    f"{provider_name} attempt {attempt+1}/{max_retries+1} failed: "
                    f"{exc!r} — retrying in {wait:.1f}s"
                )
                time.sleep(wait)
                retries += 1
            else:
                log.error(f"{provider_name} failed after {attempt+1} attempts: {exc!r}")
                break

    return None, retries, last_error


class LLMClient:
    """
    Multi-provider LLM client with retry and fallback.

    Usage:
        client = LLMClient(cfg)
        response = client.generate(prompt)
        print(response.text)
    """

    def __init__(self, cfg: dict):
        gen_cfg = cfg.get("generation", {})

        self.primary_provider = gen_cfg.get("primary_provider", "groq")
        self.primary_model = gen_cfg.get("primary_model", GROQ_DEFAULT_MODEL)
        self.fallback_provider = gen_cfg.get("fallback_provider", "openai")
        self.fallback_model = gen_cfg.get("fallback_model", OPENAI_DEFAULT_MODEL)
        self.fallback_enabled = gen_cfg.get("fallback_enabled", True)

        self.max_tokens = gen_cfg.get("max_tokens", 300)
        self.temperature = gen_cfg.get("temperature", 0.1)
        self.timeout = gen_cfg.get("timeout_seconds", HARD_TIMEOUT_SECONDS)
        self.max_retries_primary = gen_cfg.get("max_retries_primary", MAX_RETRIES_PRIMARY)
        self.max_retries_fallback = gen_cfg.get("max_retries_fallback", MAX_RETRIES_FALLBACK)

    def generate(self, prompt: str) -> LLMResponse:
        """
        Generate a completion for the given prompt.

        Tries primary provider first with retry. If all attempts fail and
        fallback is enabled, tries fallback provider. If both fail, returns
        a structured error response — caller handles gracefully.
        """
        t0 = time.time()

        primary_caller = _PROVIDER_CALLERS.get(self.primary_provider)
        if primary_caller is None:
            return self._error_response(
                f"Unknown primary provider: {self.primary_provider}",
                t0,
            )

        result, retries, error = _call_with_retry(
            caller=primary_caller,
            prompt=prompt,
            model=self.primary_model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout,
            max_retries=self.max_retries_primary,
            provider_name=self.primary_provider,
        )

        if result is not None:
            return LLMResponse(
                text=result["text"],
                provider=self.primary_provider,
                model=self.primary_model,
                prompt_tokens=result.get("prompt_tokens", 0),
                completion_tokens=result.get("completion_tokens", 0),
                latency_ms=round((time.time() - t0) * 1000, 1),
                retries_used=retries,
                success=True,
            )

        if not self.fallback_enabled:
            return self._error_response(
                f"Primary provider failed and fallback disabled: {error}",
                t0,
                retries_used=retries,
            )

        log.warning(
            f"Primary ({self.primary_provider}) exhausted — trying fallback ({self.fallback_provider})"
        )

        fallback_caller = _PROVIDER_CALLERS.get(self.fallback_provider)
        if fallback_caller is None:
            return self._error_response(
                f"Unknown fallback provider: {self.fallback_provider}",
                t0,
                retries_used=retries,
            )

        fb_result, fb_retries, fb_error = _call_with_retry(
            caller=fallback_caller,
            prompt=prompt,
            model=self.fallback_model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout,
            max_retries=self.max_retries_fallback,
            provider_name=self.fallback_provider,
        )

        total_retries = retries + fb_retries

        if fb_result is not None:
            log.info(f"Fallback ({self.fallback_provider}) succeeded after {total_retries} total retries")
            return LLMResponse(
                text=fb_result["text"],
                provider=self.fallback_provider,
                model=self.fallback_model,
                prompt_tokens=fb_result.get("prompt_tokens", 0),
                completion_tokens=fb_result.get("completion_tokens", 0),
                latency_ms=round((time.time() - t0) * 1000, 1),
                retries_used=total_retries,
                success=True,
            )

        return self._error_response(
            f"Both providers failed. Primary: {error} | Fallback: {fb_error}",
            t0,
            retries_used=total_retries,
        )

    @staticmethod
    def _error_response(
        error: str,
        t0: float,
        retries_used: int = 0,
    ) -> LLMResponse:
        log.error(f"LLMClient error: {error}")
        return LLMResponse(
            text="",
            provider="none",
            model="none",
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=round((time.time() - t0) * 1000, 1),
            retries_used=retries_used,
            error=error,
            success=False,
        )


_client: Optional[LLMClient] = None


def get_client(cfg: Optional[dict] = None) -> LLMClient:
    global _client
    if _client is None:
        # Load .env values (e.g., GROQ_API_KEY) when available.
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:
            pass
        if cfg is None:
            from pathlib import Path
            import yaml
            with open("retrieval/configs/retrieval_config.yaml", "r") as f:
                cfg = yaml.safe_load(f)
        _client = LLMClient(cfg)
    return _client
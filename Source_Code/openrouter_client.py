"""Minimal OpenRouter client wrapper using the stdlib."""

from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT_SECONDS = 45
DEFAULT_MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS: Sequence[float] = (1.0, 2.0)
RETRYABLE_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}
_RETRY_AFTER_CAP_SECONDS = 60.0  # never wait longer than this, even if Retry-After says so


@dataclass
class OpenRouterResponse:
    content: str
    model: str
    usage: Dict[str, Any]
    raw: Dict[str, Any]
    attempts: int = 1


class OpenRouterError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        attempts: int = 1,
        retryable: bool = False,
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.retryable = retryable
        self.status_code = status_code


class OpenRouterClient:
    """Small wrapper to keep API calls deterministic and easy to log."""

    def __init__(self, api_key: Optional[str]):
        if not api_key:
            raise OpenRouterError("Missing OPENROUTER_API_KEY")
        self.api_key = api_key
        self.timeout_seconds = DEFAULT_TIMEOUT_SECONDS
        self.max_retries = DEFAULT_MAX_RETRIES

    def _retry_delay(self, attempt: int) -> float:
        if not RETRY_BACKOFF_SECONDS:
            return 0.0
        index = min(max(0, attempt - 1), len(RETRY_BACKOFF_SECONDS) - 1)
        return float(RETRY_BACKOFF_SECONDS[index])

    def chat_completion(
        self,
        model: str,
        messages,
        *,
        temperature: float,
        max_tokens: int = 200,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> OpenRouterResponse:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra_body:
            payload.update(extra_body)
        max_attempts = self.max_retries + 1
        for attempt in range(1, max_attempts + 1):
            req = Request(
                OPENROUTER_API_URL,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/",
                    "X-Title": "mud-poc",
                },
                data=json.dumps(payload).encode("utf-8"),
            )
            try:
                with urlopen(req, timeout=self.timeout_seconds) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))

                choices = raw.get("choices", [])
                if not choices:
                    raise OpenRouterError(
                        f"No completion returned from model {model}: {raw}",
                        attempts=attempt,
                        retryable=True,
                    )

                content = choices[0]["message"].get("content", "") or ""
                usage = raw.get("usage", {})
                return OpenRouterResponse(
                    content=content,
                    model=raw.get("model", model),
                    usage=usage,
                    raw=raw,
                    attempts=attempt,
                )
            except HTTPError as exc:
                body = ""
                try:
                    body = exc.read().decode("utf-8")
                except Exception:
                    body = str(exc)
                retryable = exc.code in RETRYABLE_HTTP_STATUS_CODES
                if retryable and attempt < max_attempts:
                    delay = self._retry_delay(attempt)
                    if exc.code == 429:
                        try:
                            retry_after = float(
                                (exc.headers or {}).get("Retry-After") or 0
                            )
                            if retry_after > 0:
                                delay = max(delay, min(retry_after, _RETRY_AFTER_CAP_SECONDS))
                        except (ValueError, TypeError, AttributeError):
                            pass
                    if delay > 0:
                        time.sleep(delay)
                    continue
                raise OpenRouterError(
                    f"OpenRouter HTTP error {exc.code} after {attempt} attempt(s): {body}",
                    attempts=attempt,
                    retryable=retryable,
                    status_code=exc.code,
                ) from exc
            except (URLError, TimeoutError, socket.timeout) as exc:
                if attempt < max_attempts:
                    delay = self._retry_delay(attempt)
                    if delay > 0:
                        time.sleep(delay)
                    continue
                raise OpenRouterError(
                    f"OpenRouter network error after {attempt} attempt(s): {exc}",
                    attempts=attempt,
                    retryable=True,
                ) from exc
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                if attempt < max_attempts:
                    delay = self._retry_delay(attempt)
                    if delay > 0:
                        time.sleep(delay)
                    continue
                raise OpenRouterError(
                    f"OpenRouter invalid JSON response after {attempt} attempt(s): {exc}",
                    attempts=attempt,
                    retryable=True,
                ) from exc
            except OpenRouterError as exc:
                if exc.retryable and attempt < max_attempts:
                    delay = self._retry_delay(attempt)
                    if delay > 0:
                        time.sleep(delay)
                    continue
                raise
            except Exception as exc:
                raise OpenRouterError(
                    f"OpenRouter unexpected error after {attempt} attempt(s): {exc}",
                    attempts=attempt,
                    retryable=False,
                ) from exc

        raise OpenRouterError(
            f"OpenRouter request for model {model} exhausted retries without a completion",
            attempts=max_attempts,
            retryable=True,
        )


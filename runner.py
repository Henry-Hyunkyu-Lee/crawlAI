from dataclasses import dataclass
from typing import Tuple

import requests

RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class RetryPolicy:
    network_retries: int = 3
    validation_retries: int = 1
    backoff_seconds: float = 1.5
    max_backoff_seconds: float = 10.0


def classify_exception(exc: Exception) -> Tuple[str, bool]:
    status_code = getattr(exc, "status_code", None)
    if status_code is None and getattr(exc, "response", None) is not None:
        status_code = getattr(exc.response, "status_code", None)

    if status_code is not None:
        code = f"{type(exc).__name__}:{status_code}"
        return code, int(status_code) in RETRYABLE_STATUS_CODES

    if isinstance(exc, (requests.Timeout, requests.ConnectionError, requests.HTTPError)):
        return type(exc).__name__, True

    name = type(exc).__name__
    # OpenAI SDK transient exception names
    if "RateLimit" in name or "Timeout" in name or "APIConnection" in name:
        return name, True

    return name, False


def backoff_seconds(policy: RetryPolicy, retry_attempt: int) -> float:
    retry_attempt = max(1, int(retry_attempt))
    delay = policy.backoff_seconds * (2 ** (retry_attempt - 1))
    return min(delay, policy.max_backoff_seconds)

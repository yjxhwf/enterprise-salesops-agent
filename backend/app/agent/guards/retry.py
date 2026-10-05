"""Transport classification only. Never inspect error text or retry model reasoning."""
from openai import APIConnectionError, APITimeoutError, APIStatusError


def transient_code(error):
    if isinstance(error, APITimeoutError):
        return "PROVIDER_TIMEOUT"
    if isinstance(error, APIConnectionError):
        return "PROVIDER_CONNECTION_ERROR"
    if isinstance(error, APIStatusError):
        if error.status_code == 429:
            return "PROVIDER_RATE_LIMIT"
        if 500 <= error.status_code <= 599:
            return "PROVIDER_SERVER_ERROR"
    return None


def backoff(retry_number):
    return min(0.5 * 2 ** (retry_number - 1), 1.0)

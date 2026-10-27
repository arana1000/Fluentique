"""Helpers for presenting OpenAI API failures consistently."""


def describe_openai_error(exc: Exception) -> str:
    """
    Convert an OpenAI SDK exception into a concise, user-safe message.

    The application should not expose raw SDK tracebacks or response bodies to
    learners, but it should distinguish common configuration/billing problems
    from temporary upstream failures.
    """
    status_code = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)
    body = getattr(exc, "body", None)

    parts = [
        str(code or ""),
        str(body or ""),
        str(exc),
    ]
    combined = " ".join(parts).lower()

    if any(
        marker in combined
        for marker in (
            "credit_balance_exhausted",
            "insufficient_quota",
            "no credits remaining",
        )
    ):
        return (
            "OpenAI API credits are exhausted. "
            "Add credits or use a funded API key to use this AI feature."
        )

    if status_code == 401 or "invalid_api_key" in combined:
        return (
            "OpenAI API authentication failed. "
            "Check the OPENAI_API_KEY configuration."
        )

    if status_code == 404 or "model_not_found" in combined:
        return (
            "The configured OpenAI model is unavailable. "
            "Check the OPENAI_MODEL configuration."
        )

    if status_code == 429:
        return (
            "OpenAI is rate-limiting requests right now. "
            "Try this AI feature again shortly."
        )

    if isinstance(status_code, int) and status_code >= 500:
        return (
            "OpenAI is temporarily unavailable. "
            "Try this AI feature again later."
        )

    return (
        "OpenAI service is currently unavailable. "
        "Check the API configuration and try again."
    )

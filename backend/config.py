import logging
import os
from pathlib import Path

from dotenv import load_dotenv


logger = logging.getLogger(__name__)


# config.py
# ----------------------
# Central configuration for the Fluentique backend.
#
# Secrets and machine-specific values are loaded from the project-root .env
# file. Non-secret runtime settings use sensible defaults where appropriate.
#
# Google Speech credentials are intentionally no longer part of the production
# runtime configuration because speech transcription now uses OpenAI.


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)


# === Environment Helpers ===

def _get_required_env(name: str) -> str:
    """
    Return a required environment variable or fail clearly at startup.
    """
    value = os.getenv(name)

    if value is None or not value.strip():
        raise RuntimeError(
            f"Required environment variable '{name}' is not set."
        )

    return value.strip()


def _get_env(
    name: str,
    default: str,
) -> str:
    """
    Return an environment variable, falling back to a non-secret default.
    """
    value = os.getenv(name)

    if value is None or not value.strip():
        return default

    return value.strip()


def _resolve_project_path(value: str) -> Path:
    """
    Resolve a configured path relative to the project root unless it is
    already absolute.
    """
    path = Path(value).expanduser()

    if path.is_absolute():
        return path

    return BASE_DIR / path


# === Firebase ===

FIREBASE_CREDENTIALS = str(
    _resolve_project_path(
        _get_required_env("FIREBASE_CREDENTIALS")
    )
)

FIREBASE_WEB_API_KEY = _get_required_env(
    "FIREBASE_WEB_API_KEY"
)

FIREBASE_STORAGE_BUCKET = _get_required_env(
    "FIREBASE_STORAGE_BUCKET"
)


# === OpenAI ===

OPENAI_API_KEY = _get_required_env(
    "OPENAI_API_KEY"
)

OPENAI_MODEL = _get_env(
    "OPENAI_MODEL",
    "gpt-4o-mini",
)

# Older local .env files from the original FYP may still specify a retired
# GPT-3.5 model. Fall back automatically so an old local config does not break
# every OpenAI-powered feature after the runtime refactor.
if OPENAI_MODEL.lower().startswith("gpt-3.5-turbo"):
    logger.warning(
        "OPENAI_MODEL=%s is no longer available; using gpt-4o-mini instead.",
        OPENAI_MODEL,
    )
    OPENAI_MODEL = "gpt-4o-mini"

OPENAI_TRANSCRIPTION_MODEL = _get_env(
    "OPENAI_TRANSCRIPTION_MODEL",
    "gpt-4o-mini-transcribe",
)


# === Amazon Polly ===

AWS_REGION = _get_env(
    "AWS_REGION",
    "eu-west-2",
)

POLLY_VOICE_ID = _get_env(
    "POLLY_VOICE_ID",
    "Celine",
)

POLLY_ENGLISH_VOICE_ID = _get_env(
    "POLLY_ENGLISH_VOICE_ID",
    "Amy",
)

POLLY_LANGUAGE_CODE = _get_env(
    "POLLY_LANGUAGE_CODE",
    "fr-FR",
)


# === Backend ===

CORS_ORIGINS = _get_env(
    "CORS_ORIGINS",
    "*",
)

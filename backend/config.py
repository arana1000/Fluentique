import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


def _get_required_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Required environment variable '{name}' is not set."
        )

    return value


FIREBASE_CREDENTIALS = str(
    BASE_DIR / _get_required_env("FIREBASE_CREDENTIALS")
)

GOOGLE_CREDS_PATH = str(
    BASE_DIR / _get_required_env("GOOGLE_APPLICATION_CREDENTIALS")
)

OPENAI_API_KEY = _get_required_env("OPENAI_API_KEY")
FIREBASE_WEB_API_KEY = _get_required_env("FIREBASE_WEB_API_KEY")


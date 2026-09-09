import logging

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from firebase_admin import auth


logger = logging.getLogger(__name__)

security = HTTPBearer()


def verify_token(token: str) -> dict:
    """Verify a Firebase ID token and return the decoded user data."""
    try:
        return auth.verify_id_token(
            token,
            clock_skew_seconds=5,
        )

    except Exception:
        logger.exception("Firebase token verification failed.")
        raise HTTPException(
            status_code=401,
            detail="Invalid Firebase token",
        ) from None


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Verify the bearer token supplied to a protected FastAPI route."""
    return verify_token(credentials.credentials)
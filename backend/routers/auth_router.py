import logging

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr

from backend.config import FIREBASE_WEB_API_KEY

# auth_router.py
# ----------------------
# This file handles all authentication-related routes for the application.
# It includes user registration, login, and password reset functionalities.
# The routes interact with Firebase Authentication to securely handle user
# credentials and sessions.
#
# Functions:
# - register_user: Registers a new user by sending credentials to Firebase.
# - login_user: Verifies user credentials and returns a session token.
# - reset_password: Sends a password reset link to the user's email.


logger = logging.getLogger(__name__)

# Create a FastAPI router for authentication-related endpoints
router = APIRouter(prefix="/auth", tags=["Auth"])

# Timeout used for requests to Firebase Authentication
REQUEST_TIMEOUT = 10


# === Request Models ===

class AuthDetails(BaseModel):
    """
    Model for user authentication requests (registration and login).
    Requires an email and password.
    """

    email: EmailStr
    password: str


class PasswordReset(BaseModel):
    """
    Model for initiating a password reset.
    Only requires a valid email address.
    """

    email: EmailStr


# === Firebase REST API Endpoints ===

FIREBASE_SIGNUP = (
    "https://identitytoolkit.googleapis.com/v1/accounts:signUp"
    f"?key={FIREBASE_WEB_API_KEY}"
)

FIREBASE_LOGIN = (
    "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
    f"?key={FIREBASE_WEB_API_KEY}"
)

FIREBASE_RESET = (
    "https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode"
    f"?key={FIREBASE_WEB_API_KEY}"
)


# === Firebase Request Helper ===

def _firebase_request(url: str, payload: dict) -> dict:
    """
    Sends a request to Firebase Authentication and handles common errors.
    """
    try:
        response = requests.post(
            url,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

    except requests.RequestException:
        logger.exception("Firebase Authentication request failed.")

        raise HTTPException(
            status_code=503,
            detail="Authentication service is temporarily unavailable.",
        ) from None

    try:
        response_data = response.json()

    except requests.JSONDecodeError:
        logger.error(
            "Firebase returned an invalid response with status %s.",
            response.status_code,
        )

        raise HTTPException(
            status_code=502,
            detail="Invalid response from authentication service.",
        ) from None

    if response.ok:
        return response_data

    error_message = (
        response_data.get("error", {}).get("message")
        or "Authentication request failed."
    )

    raise HTTPException(
        status_code=400,
        detail=error_message,
    )


# === Registration Route ===

@router.post("/register")
def register_user(data: AuthDetails) -> dict:
    """
    Registers a new user with Firebase Authentication.
    Accepts an email and password, and returns an ID token if successful.
    """
    payload = {
        "email": str(data.email),
        "password": data.password,
        "returnSecureToken": True,
    }

    response = _firebase_request(
        FIREBASE_SIGNUP,
        payload,
    )

    return {
        "msg": "User registered",
        "idToken": response["idToken"],
    }


# === Login Route ===

@router.post("/login")
def login_user(data: AuthDetails) -> dict:
    """
    Logs in an existing user by verifying credentials with Firebase.
    Returns both an ID token and a refresh token upon success.
    """
    payload = {
        "email": str(data.email),
        "password": data.password,
        "returnSecureToken": True,
    }

    response = _firebase_request(
        FIREBASE_LOGIN,
        payload,
    )

    return {
        "msg": "Login successful",
        "idToken": response["idToken"],
        "refreshToken": response["refreshToken"],
    }


# === Password Reset Route ===

@router.post("/reset")
def reset_password(data: PasswordReset) -> dict:
    """
    Sends a password reset email to the user's registered email address.
    Uses Firebase's password reset API endpoint.
    """
    payload = {
        "requestType": "PASSWORD_RESET",
        "email": str(data.email),
    }

    _firebase_request(
        FIREBASE_RESET,
        payload,
    )

    return {
        "msg": f"Password reset email sent to {data.email}",
    }
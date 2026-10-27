import logging
import uuid
from urllib.parse import quote

import firebase_admin
from firebase_admin import credentials, firestore, storage

from backend.config import (
    FIREBASE_CREDENTIALS,
    FIREBASE_STORAGE_BUCKET,
)


# firebase_utils.py
# ----------------------
# Firebase initialisation and shared Firestore / Storage helpers for Fluentique.
#
# Credentials and bucket configuration are loaded centrally through
# backend.config so this module contains no machine-specific paths or secrets.


logger = logging.getLogger(__name__)


# === Firebase Initialisation ===

def _initialise_firebase():
    """
    Return the existing Firebase app or initialise it if required.
    """
    try:
        return firebase_admin.get_app()

    except ValueError:
        credential = credentials.Certificate(
            FIREBASE_CREDENTIALS
        )

        return firebase_admin.initialize_app(
            credential,
            {
                "storageBucket": FIREBASE_STORAGE_BUCKET,
            },
        )


# Initialise Firebase once and create reusable clients.
firebase_app = _initialise_firebase()
db = firestore.client(app=firebase_app)
bucket = storage.bucket(app=firebase_app)


# === User Data ===

def update_user_data(
    uid: str,
    email: str,
    xp: int,
    league: str,
    profile_picture_url: str | None = None,
    streak: int | None = None,
    last_completed: str | None = None,
) -> None:
    """
    Create or update a user's Firestore document.

    Existing fields not included in this update are preserved. The profile
    picture URL is updated only when a value is explicitly supplied.
    """
    try:
        user_ref = db.collection(
            "users"
        ).document(uid)

        data = {
            "uid": uid,
            "email": email,
            "xp": xp,
            "league": league,
        }

        if profile_picture_url is not None:
            data["profile_picture_url"] = profile_picture_url

        if streak is not None:
            data["streak"] = streak

        if last_completed is not None:
            data["last_completed"] = last_completed

        user_ref.set(
            data,
            merge=True,
        )

    except Exception:
        logger.exception(
            "Failed to update Firebase data for user %s.",
            uid,
        )


def get_all_users() -> list[dict]:
    """
    Retrieve all user documents.

    Used by the Fluentique leaderboard.
    """
    try:
        users_ref = db.collection("users")
        return [
            doc.to_dict()
            for doc in users_ref.stream()
        ]

    except Exception:
        logger.exception(
            "Failed to retrieve users from Firestore."
        )
        return []


# === Profile Images ===

def upload_profile_image(
    uid: str,
    file_path: str,
) -> str | None:
    """
    Upload a profile image to Firebase Storage.

    Returns the Firebase download URL on success or None on failure.
    """
    try:
        filename = (
            f"profile_pictures/{uid}/{uuid.uuid4()}.jpg"
        )

        blob = bucket.blob(filename)
        blob.upload_from_filename(file_path)

        token = str(uuid.uuid4())

        blob.metadata = {
            "firebaseStorageDownloadTokens": token,
        }
        blob.patch()

        encoded_filename = quote(
            filename,
            safe="",
        )

        return (
            "https://firebasestorage.googleapis.com/v0/b/"
            f"{bucket.name}/o/{encoded_filename}"
            f"?alt=media&token={token}"
        )

    except Exception:
        logger.exception(
            "Failed to upload profile image for user %s.",
            uid,
        )
        return None


def update_profile_picture_url(
    uid: str,
    url: str,
) -> None:
    """
    Update a user's profile picture URL in Firestore.
    """
    try:
        user_ref = db.collection(
            "users"
        ).document(uid)

        user_ref.update(
            {
                "profile_picture_url": url,
            }
        )

    except Exception:
        logger.exception(
            "Failed to update profile picture URL for user %s.",
            uid,
        )


# === Profile Details ===

def update_username(
    uid: str,
    username: str,
) -> None:
    """
    Update a user's username.
    """
    try:
        user_ref = db.collection(
            "users"
        ).document(uid)

        user_ref.update(
            {
                "username": username,
            }
        )

    except Exception:
        logger.exception(
            "Failed to update username for user %s.",
            uid,
        )


def get_user_data(
    uid: str,
) -> dict | None:
    """
    Retrieve a user's Firestore document.

    Returns the user's data or None if the document does not exist
    or cannot be retrieved.
    """
    try:
        user_ref = db.collection(
            "users"
        ).document(uid)

        doc = user_ref.get()

        if not doc.exists:
            return None

        return doc.to_dict()

    except Exception:
        logger.exception(
            "Failed to retrieve Firebase data for user %s.",
            uid,
        )
        return None

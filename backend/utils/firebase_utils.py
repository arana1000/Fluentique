import logging
import os
import uuid
from urllib.parse import quote

import firebase_admin
from firebase_admin import credentials, firestore, storage

from backend.config import FIREBASE_CREDENTIALS


logger = logging.getLogger(__name__)

FIREBASE_STORAGE_BUCKET = os.getenv(
    "FIREBASE_STORAGE_BUCKET",
    "fluentiqueapp.firebasestorage.app",
)


def _initialise_firebase():
    """
    Return the existing Firebase app or initialise it if required.
    """
    try:
        return firebase_admin.get_app()
    except ValueError:
        cred = credentials.Certificate(FIREBASE_CREDENTIALS)

        return firebase_admin.initialize_app(
            cred,
            {
                "storageBucket": FIREBASE_STORAGE_BUCKET,
            },
        )


# Initialise Firebase once and create reusable clients.
firebase_app = _initialise_firebase()
db = firestore.client(app=firebase_app)
bucket = storage.bucket(app=firebase_app)


def update_user_data(
    uid: str,
    email: str,
    xp: int,
    league: str,
    profile_picture_url: str = "",
) -> None:
    """
    Create or update a user's Firestore document.

    Existing fields not included in this update are preserved.
    """
    try:
        user_ref = db.collection("users").document(uid)

        user_ref.set(
            {
                "uid": uid,
                "email": email,
                "xp": xp,
                "league": league,
                "profile_picture_url": profile_picture_url,
            },
            merge=True,
        )

    except Exception:
        logger.exception("Failed to update Firebase data for user %s.", uid)


def get_all_users() -> list[dict]:
    """
    Retrieve all user documents.

    Used by the Fluentique leaderboard.
    """
    try:
        users_ref = db.collection("users")
        return [doc.to_dict() for doc in users_ref.stream()]

    except Exception:
        logger.exception("Failed to retrieve users from Firestore.")
        return []


def upload_profile_image(uid: str, file_path: str) -> str | None:
    """
    Upload a profile image to Firebase Storage.

    Returns the Firebase download URL on success or None on failure.
    """
    try:
        filename = f"profile_pictures/{uid}/{uuid.uuid4()}.jpg"

        blob = bucket.blob(filename)
        blob.upload_from_filename(file_path)

        token = str(uuid.uuid4())

        blob.metadata = {
            "firebaseStorageDownloadTokens": token,
        }
        blob.patch()

        encoded_filename = quote(filename, safe="")

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


def update_profile_picture_url(uid: str, url: str) -> None:
    """
    Update a user's profile picture URL in Firestore.
    """
    try:
        user_ref = db.collection("users").document(uid)
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


def update_username(uid: str, username: str) -> None:
    """
    Update a user's username in Firestore.
    """
    try:
        user_ref = db.collection("users").document(uid)
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


def get_user_data(uid: str) -> dict | None:
    """
    Retrieve a user's Firestore document.

    Returns the user's data or None if the document does not exist
    or cannot be retrieved.
    """
    try:
        user_ref = db.collection("users").document(uid)
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
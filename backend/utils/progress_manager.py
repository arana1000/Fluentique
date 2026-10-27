import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

from kivy.app import App

from backend.utils.auth_utils import verify_token
from backend.utils.firebase_utils import get_user_data, update_user_data


# progress_manager.py
# ----------------------
# Manages Fluentique's local learning progress and synchronises the user's
# XP, league, streak, and last activity date with Firebase.


logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROGRESS_FILE = DATA_DIR / "progress.json"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


class ProgressManager:
    def __init__(self) -> None:
        self.uid: str | None = None
        self.progress = self._default_progress()

    @staticmethod
    def _default_progress() -> dict[str, Any]:
        """Return a fresh default progress record."""
        return {
            "xp": 0,
            "sentences_completed": 0,
            "speech_completed": 0,
            "quiz_completed": 0,
            "streak": 0,
            "last_completed": "",
        }

    def load_user(self, uid: str) -> None:
        """
        Load progress for the currently authenticated user.

        The in-memory state is reset first so progress from a previously
        logged-in user can never leak into a new user's session.
        """
        self.uid = uid
        self.progress = self._default_progress()
        self._load()

    def _load_progress_file(self) -> dict:
        """Read all locally cached progress records."""
        if not PROGRESS_FILE.exists():
            return {}

        try:
            with PROGRESS_FILE.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

            return data if isinstance(data, dict) else {}

        except (OSError, json.JSONDecodeError):
            logger.exception(
                "Failed to read local progress file."
            )
            return {}

    def _load(self) -> None:
        """Load the active user's progress from local cache or Firebase."""
        if self.uid is None:
            logger.warning(
                "Cannot load progress without a user ID."
            )
            return

        data = self._load_progress_file()

        local_progress = data.get(
            self.uid
        )

        if isinstance(local_progress, dict):
            self.progress = {
                **self._default_progress(),
                **local_progress,
            }
            return

        user_data = get_user_data(
            self.uid
        )

        if user_data:
            self.progress = {
                **self._default_progress(),
                "xp": user_data.get("xp", 0),
                "streak": user_data.get("streak", 0),
                "last_completed": user_data.get(
                    "last_completed",
                    "",
                ),
            }

        # Cache the loaded/default progress locally for this user.
        self._save()

    def _save_all(
        self,
        data: dict,
    ) -> None:
        """Write all local progress records to disk."""
        try:
            DATA_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            with PROGRESS_FILE.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    data,
                    file,
                    indent=2,
                )

        except OSError:
            logger.exception(
                "Failed to save local progress data."
            )

    def _save(self) -> None:
        """Save the active user's progress to the local cache."""
        if self.uid is None:
            logger.warning(
                "Cannot save progress without a user ID."
            )
            return

        data = self._load_progress_file()
        data[self.uid] = self.progress

        self._save_all(
            data
        )

    def _sync_if_authenticated(self) -> None:
        """Synchronise progress when an authenticated Kivy app is available."""
        app = App.get_running_app()
        token = getattr(
            app,
            "token",
            None,
        ) if app else None

        if token:
            self.sync_to_firebase(
                token
            )
        else:
            logger.warning(
                "Firebase sync skipped because no token is available."
            )

    def add_xp(
        self,
        xp: int,
    ) -> None:
        """Add experience points, update the streak, and save progress."""
        if self.uid is None:
            logger.warning(
                "Cannot add XP without an active user."
            )
            return

        self.progress["xp"] += xp
        self._update_streak()
        self._save()
        self._sync_if_authenticated()

    def increment_sentence(self) -> None:
        """Record a completed sentence exercise and award 10 XP."""
        self.progress["sentences_completed"] += 1
        self.add_xp(10)

    def increment_speech(self) -> None:
        """Record a completed speech exercise and award 5 XP."""
        self.progress["speech_completed"] += 1
        self.add_xp(5)

    def increment_quiz(self) -> None:
        """Record a completed quiz question and award 10 XP."""
        self.progress["quiz_completed"] += 1
        self.add_xp(10)

    def _update_streak(self) -> None:
        """Update the learning streak based on the last activity date."""
        today = date.today()
        last_completed = self.progress.get(
            "last_completed",
            "",
        )

        if last_completed:
            try:
                last_date = datetime.strptime(
                    last_completed,
                    "%Y-%m-%d",
                ).date()

                days_since_last = (
                    today - last_date
                ).days

                if days_since_last == 1:
                    self.progress["streak"] += 1

                elif days_since_last > 1:
                    self.progress["streak"] = 1

                elif days_since_last < 0:
                    logger.warning(
                        "Stored progress date is in the future: %s",
                        last_completed,
                    )

            except (TypeError, ValueError):
                self.progress["streak"] = 1

        else:
            self.progress["streak"] = 1

        self.progress["last_completed"] = (
            today.isoformat()
        )

    def update_streak(self) -> None:
        """Manually trigger a streak update and synchronise it."""
        if self.uid is None:
            logger.warning(
                "Cannot update streak without an active user."
            )
            return

        self._update_streak()
        self._save()
        self._sync_if_authenticated()

    def get_progress(self) -> dict[str, Any]:
        """Return the current in-memory progress state."""
        return self.progress

    def sync_to_firebase(
        self,
        token: str,
    ) -> None:
        """Synchronise progress and league information with Firestore."""
        try:
            user_info = verify_token(
                token
            )

            uid = user_info["uid"]
            email = user_info.get(
                "email",
                "unknown@example.com",
            )
            xp = self.progress.get(
                "xp",
                0,
            )

            if xp >= 400:
                league = "Gold"

            elif xp >= 200:
                league = "Silver"

            else:
                league = "Bronze"

            update_user_data(
                uid=uid,
                email=email,
                xp=xp,
                league=league,
                streak=self.progress.get(
                    "streak",
                    0,
                ),
                last_completed=self.progress.get(
                    "last_completed",
                    "",
                ),
            )

        except Exception:
            logger.exception(
                "Failed to synchronise progress with Firebase."
            )

    def reset_progress(self) -> None:
        """
        Clear the active in-memory session without deleting saved progress.

        The frontend calls this during logout. Persisting a zeroed record here
        would erase the user's locally cached progress, so logout only detaches
        the current user and restores a clean in-memory state.
        """
        self.uid = None
        self.progress = self._default_progress()

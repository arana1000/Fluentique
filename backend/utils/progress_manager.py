import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

from kivy.app import App

from backend.utils.auth_utils import verify_token
from backend.utils.firebase_utils import get_user_data, update_user_data


logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROGRESS_FILE = DATA_DIR / "progress.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)


class ProgressManager:
    def __init__(self) -> None:
        self.uid: str | None = None
        self.progress = self._default_progress()

    @staticmethod
    def _default_progress() -> dict[str, Any]:
        return {
            "xp": 0,
            "sentences_completed": 0,
            "speech_completed": 0,
            "quiz_completed": 0,
            "streak": 0,
            "last_completed": "",
        }

    def load_user(self, uid: str) -> None:
        """Load progress for the currently authenticated user."""
        self.uid = uid
        self._load()

    def _load_progress_file(self) -> dict:
        if not PROGRESS_FILE.exists():
            return {}

        try:
            with PROGRESS_FILE.open("r", encoding="utf-8") as file:
                data = json.load(file)

            return data if isinstance(data, dict) else {}

        except (OSError, json.JSONDecodeError):
            logger.exception("Failed to read local progress file.")
            return {}

    def _load(self) -> None:
        if self.uid is None:
            logger.warning("Cannot load progress without a user ID.")
            return

        data = self._load_progress_file()

        if self.uid in data and isinstance(data[self.uid], dict):
            self.progress = {
                **self._default_progress(),
                **data[self.uid],
            }
            return

        user_data = get_user_data(self.uid)

        if user_data:
            self.progress = {
                **self._default_progress(),
                "xp": user_data.get("xp", 0),
                "streak": user_data.get("streak", 0),
                "last_completed": user_data.get("last_completed", ""),
            }

        self._save()

    def _save_all(self, data: dict) -> None:
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)

            with PROGRESS_FILE.open("w", encoding="utf-8") as file:
                json.dump(data, file, indent=2)

        except OSError:
            logger.exception("Failed to save local progress data.")

    def _save(self) -> None:
        if self.uid is None:
            logger.warning("Cannot save progress without a user ID.")
            return

        data = self._load_progress_file()
        data[self.uid] = self.progress

        self._save_all(data)

    def add_xp(self, xp: int) -> None:
        """Add experience points and synchronise progress."""
        self.progress["xp"] += xp
        self._update_streak()
        self._save()

        app = App.get_running_app()
        token = getattr(app, "token", None)

        if token:
            self.sync_to_firebase(token)
        else:
            logger.warning("Firebase sync skipped because no token is available.")

    def increment_sentence(self) -> None:
        self.progress["sentences_completed"] += 1
        self.add_xp(10)

    def increment_speech(self) -> None:
        self.progress["speech_completed"] += 1
        self.add_xp(5)

    def increment_quiz(self) -> None:
        self.progress["quiz_completed"] += 1
        self.add_xp(10)

    def _update_streak(self) -> None:
        """Update the learning streak based on the last activity date."""
        today = date.today()
        last_completed = self.progress.get("last_completed", "")

        if last_completed:
            try:
                last_date = datetime.strptime(
                    last_completed,
                    "%Y-%m-%d",
                ).date()

                days_since_last = (today - last_date).days

                if days_since_last == 1:
                    self.progress["streak"] += 1
                elif days_since_last > 1:
                    self.progress["streak"] = 1

            except (TypeError, ValueError):
                self.progress["streak"] = 1

        else:
            self.progress["streak"] = 1

        self.progress["last_completed"] = today.isoformat()

    def update_streak(self) -> None:
        """Manually trigger a streak update."""
        self._update_streak()
        self._save()

    def get_progress(self) -> dict[str, Any]:
        return self.progress

    def sync_to_firebase(self, token: str) -> None:
        """Synchronise XP and league information with Firestore."""
        try:
            user_info = verify_token(token)

            uid = user_info["uid"]
            email = user_info.get("email", "unknown@example.com")
            xp = self.progress.get("xp", 0)

            if xp >= 400:
                league = "Gold"
            elif xp >= 200:
                league = "Silver"
            else:
                league = "Bronze"

            update_user_data(uid, email, xp, league)

        except Exception:
            logger.exception("Failed to synchronise progress with Firebase.")

    def reset_progress(self) -> None:
        """Reset progress for the current user."""
        self.progress = self._default_progress()
        self._save()
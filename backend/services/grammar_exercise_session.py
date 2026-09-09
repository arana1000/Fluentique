import json
import logging
import random
from pathlib import Path


# grammar_exercise_session.py
# ----------------------
# This file manages the session lifecycle for grammar exercises.
# It handles the initialization of grammar practice sessions, tracking progress
# through various exercises, and providing feedback after each user input.
#
# Functions:
# - start_session: Initializes a new grammar exercise session for the user.
# - track_progress: Tracks user progress through exercises.
# - provide_feedback: Gives feedback on the grammar exercise result.


logger = logging.getLogger(__name__)

SESSION_LENGTH = 5


# === Load grammar translation exercises from JSON file ===

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXERCISE_FILE = DATA_DIR / "beginner_translation_exercises.json"

try:
    exercise_data = json.loads(
        EXERCISE_FILE.read_text(encoding="utf-8")
    )
except (OSError, json.JSONDecodeError) as exc:
    raise RuntimeError(
        f"Unable to load grammar exercises from {EXERCISE_FILE}."
    ) from exc

if not isinstance(exercise_data, list) or len(exercise_data) < SESSION_LENGTH:
    raise RuntimeError(
        f"Grammar exercise data must contain at least {SESSION_LENGTH} exercises."
    )


# === Temporary user session store (can be replaced with database) ===

user_sessions: dict[str, dict] = {}


# === Fetch a new translation exercise for a user ===

def get_new_exercise(username: str) -> dict:
    """
    Returns the current exercise for a user, creating a new session if needed.
    """
    session = user_sessions.get(username)

    if not session:
        # Pick 5 unique random exercises per session
        selected = random.sample(
            range(len(exercise_data)),
            SESSION_LENGTH,
        )

        user_sessions[username] = {
            "indices": selected,
            "current": 0,
            "responses": [],
        }

        session = user_sessions[username]

    index = session["indices"][session["current"]]

    return {
        "index": index,
        "english": exercise_data[index]["english"],
        "french": exercise_data[index]["french"],
    }


# === Submit a user's translated sentence and return feedback ===

def submit_translation(
    username: str,
    user_input: str,
) -> dict:
    """
    Records a grammar exercise answer and advances the user's session.
    """
    session = user_sessions.get(username)

    if not session:
        return {
            "message": "No active session.",
        }

    current_position = session["current"]
    exercise_index = session["indices"][current_position]
    correct_french = exercise_data[exercise_index]["french"]

    logger.debug(
        "Submitting grammar exercise %s for user %s.",
        exercise_index,
        username,
    )

    # Basic string comparison
    is_correct = (
        user_input.strip().lower()
        == correct_french.strip().lower()
    )

    # Placeholder feedback
    if is_correct:
        explanation = "No mistakes - well done!"
    else:
        explanation = "Sentence did not match exactly."

    # Record the result
    session["responses"].append(
        {
            "prompt": exercise_data[exercise_index]["english"],
            "expected": correct_french,
            "user": user_input,
            "correct": is_correct,
            "feedback": explanation,
        }
    )

    # Move to next question
    session["current"] += 1

    # If session is complete (5 questions), return results and end session
    if session["current"] >= SESSION_LENGTH:
        results = session["responses"]

        del user_sessions[username]  # Session cleanup

        return {
            "message": "Completed 5 exercises.",
            "results": results,
        }

    # Otherwise, return the next prompt
    next_index = session["indices"][session["current"]]

    return {
        "message": "Answer submitted.",
        "next": exercise_data[next_index]["english"],
        "responses": session["responses"],
    }
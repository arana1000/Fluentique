import json
import logging
import random
from pathlib import Path

from backend.services.grammar_feedback import generate_feedback


# grammar_exercise_session.py
# ----------------------
# This file manages the session lifecycle for grammar exercises.
# It loads curated translation exercises, creates short practice sessions,
# tracks progress through each session, and stores feedback for each answer.
#
# The exercise content remains local and curated. OpenAI is used only to
# generate useful feedback when a learner submits an incorrect sentence.
#
# Functions:
# - get_new_exercise: Creates or resumes a grammar exercise session.
# - submit_translation: Validates an answer, generates feedback, and advances
#   the session.


logger = logging.getLogger(__name__)

SESSION_LENGTH = 5


# === Load grammar translation exercises from JSON file ===

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXERCISE_FILE = DATA_DIR / "beginner_translation_exercises.json"


def _load_exercise_data() -> list[dict]:
    """
    Load and validate the curated grammar exercise dataset.
    """
    try:
        data = json.loads(
            EXERCISE_FILE.read_text(
                encoding="utf-8"
            )
        )

    except (OSError, json.JSONDecodeError) as exc:
        logger.exception(
            "Failed to load grammar exercises from %s.",
            EXERCISE_FILE,
        )

        raise RuntimeError(
            f"Unable to load grammar exercises from {EXERCISE_FILE}."
        ) from exc

    if not isinstance(data, list) or len(data) < SESSION_LENGTH:
        raise RuntimeError(
            f"Grammar exercise data must contain at least "
            f"{SESSION_LENGTH} exercises."
        )

    for index, exercise in enumerate(data):
        if (
            not isinstance(exercise, dict)
            or not isinstance(exercise.get("english"), str)
            or not exercise["english"].strip()
            or not isinstance(exercise.get("french"), str)
            or not exercise["french"].strip()
        ):
            raise RuntimeError(
                f"Grammar exercise at index {index} is invalid."
            )

    return data


exercise_data = _load_exercise_data()


# === Temporary User Session Store ===

# Sessions are intentionally kept in memory for the current desktop
# application. A persistent/shared store would be needed if the backend were
# later deployed across multiple server processes or instances.
user_sessions: dict[str, dict] = {}


# === Session Helpers ===

def _normalise_sentence(text: str) -> str:
    """
    Normalise surrounding/repeated whitespace and letter case for comparison.
    """
    return " ".join(
        text.strip().split()
    ).casefold()


def _get_session_exercise(
    session: dict,
) -> tuple[int, dict]:
    """
    Resolve the session's current position to its selected exercise.
    """
    current_position = session.get(
        "current"
    )
    indices = session.get(
        "indices"
    )
    responses = session.get(
        "responses"
    )

    if (
        not isinstance(current_position, int)
        or not isinstance(indices, list)
        or not isinstance(responses, list)
        or current_position < 0
        or current_position >= len(indices)
    ):
        raise ValueError(
            "Grammar exercise session is invalid."
        )

    exercise_index = indices[
        current_position
    ]

    if (
        not isinstance(exercise_index, int)
        or exercise_index < 0
        or exercise_index >= len(exercise_data)
    ):
        raise ValueError(
            "Grammar exercise session references invalid exercise data."
        )

    return (
        exercise_index,
        exercise_data[exercise_index],
    )


# === Fetch a New Translation Exercise for a User ===

def get_new_exercise(
    user_id: str,
) -> dict:
    """
    Return the current exercise for a user, creating a new session if needed.

    The French sentence is intentionally returned because the current Kivy
    frontend uses it to construct and shuffle the sentence-building word tiles.
    """
    session = user_sessions.get(
        user_id
    )

    if not session:
        # Pick unique random exercises for this session.
        selected = random.sample(
            range(len(exercise_data)),
            SESSION_LENGTH,
        )

        session = {
            "indices": selected,
            "current": 0,
            "responses": [],
        }

        user_sessions[user_id] = session

    try:
        exercise_index, exercise = _get_session_exercise(
            session
        )

    except ValueError as exc:
        logger.exception(
            "Invalid grammar session for user %s.",
            user_id,
        )

        # Remove corrupted session state so the next request can start fresh.
        user_sessions.pop(
            user_id,
            None,
        )

        raise RuntimeError(
            "Grammar exercise session is invalid."
        ) from exc

    return {
        "index": exercise_index,
        "english": exercise["english"],
        "french": exercise["french"],
    }


# === Submit a User's Translation and Return Feedback ===

def submit_translation(
    user_id: str,
    user_input: str,
) -> dict:
    """
    Record a grammar exercise answer, generate feedback, and advance the session.
    """
    session = user_sessions.get(
        user_id
    )

    if not session:
        raise ValueError(
            "No active grammar exercise session."
        )

    submitted_text = user_input.strip()

    if not submitted_text:
        raise ValueError(
            "No answer submitted."
        )

    try:
        exercise_index, exercise = _get_session_exercise(
            session
        )

    except ValueError as exc:
        logger.exception(
            "Invalid grammar session while submitting for user %s.",
            user_id,
        )

        user_sessions.pop(
            user_id,
            None,
        )

        raise ValueError(
            "Grammar exercise session is invalid."
        ) from exc

    correct_french = exercise[
        "french"
    ]

    logger.debug(
        "Submitting grammar exercise %s for user %s.",
        exercise_index,
        user_id,
    )

    is_correct = (
        _normalise_sentence(
            submitted_text
        )
        == _normalise_sentence(
            correct_french
        )
    )

    if is_correct:
        explanation = (
            "No mistakes - well done!"
        )

    else:
        # Compare the learner's attempt with the curated target sentence and
        # provide a concise explanation through the OpenAI-backed feedback
        # service.
        explanation = generate_feedback(
            original=submitted_text,
            corrected=correct_french,
        )

    # Record the result.
    session["responses"].append(
        {
            "prompt": exercise["english"],
            "expected": correct_french,
            "user": submitted_text,
            "correct": is_correct,
            "feedback": explanation,
        }
    )

    # Move to the next question.
    session["current"] += 1

    session_length = len(
        session["indices"]
    )

    # If the session is complete, return results and remove session state.
    if session["current"] >= session_length:
        results = list(
            session["responses"]
        )

        user_sessions.pop(
            user_id,
            None,
        )

        return {
            "message": (
                f"Completed {session_length} exercises."
            ),
            "results": results,
        }

    # Otherwise, return the next prompt and the accumulated responses.
    try:
        _, next_exercise = _get_session_exercise(
            session
        )

    except ValueError as exc:
        logger.exception(
            "Invalid next grammar exercise for user %s.",
            user_id,
        )

        responses = list(
            session["responses"]
        )

        user_sessions.pop(
            user_id,
            None,
        )

        raise ValueError(
            "Grammar exercise session became invalid."
        ) from exc

    return {
        "message": "Answer submitted.",
        "next": next_exercise["english"],
        "responses": list(
            session["responses"]
        ),
    }

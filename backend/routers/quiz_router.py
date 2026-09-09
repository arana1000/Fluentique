import json
import logging
import random
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.utils.auth_utils import get_current_user


# quiz_router.py
# ----------------------
# This file handles the routing for the quiz section of the app.
# It provides functionality for fetching quiz questions, validating user responses,
# and scoring. The quiz dynamically adjusts the difficulty level based on
# the user’s performance, and it logs progress for further evaluation.
#
# Functions:
# - fetch_quiz: Retrieves quiz questions and possible answers.
# - submit_answer: Validates the user’s answer and provides immediate feedback.
# - track_progress: Tracks the user's score and adjusts difficulty based on their answers.


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/quiz",
    tags=["Quiz"],
)


# === Quiz Data ===

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
QUIZ_FILE = DATA_DIR / "french_quiz_data_aws.json"


def _load_quiz_data() -> list[dict]:
    """
    Loads quiz questions from the bundled JSON data file.
    """
    try:
        with QUIZ_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)

    except (OSError, json.JSONDecodeError) as exc:
        logger.exception(
            "Failed to load quiz data from %s.",
            QUIZ_FILE,
        )
        raise RuntimeError(
            f"Unable to load quiz data from {QUIZ_FILE}."
        ) from exc

    if not isinstance(data, list) or not data:
        raise RuntimeError(
            "Quiz data must contain a non-empty list of questions."
        )

    return data


quiz_data = _load_quiz_data()


# === Request Models ===

class QuizResponse(BaseModel):
    """
    Request body schema for submitting a quiz answer.
    """

    selected_option: str


# === Quiz Question Route ===

@router.get("/question")
def get_quiz_question(
    user=Depends(get_current_user),
):
    """
    Returns a randomly selected quiz question.

    Response:
        - question: the correct French word (used for TTS playback)
        - options: list of four possible answers
    """
    question = random.choice(quiz_data)

    return {
        "question": question["correct"],
        "options": question["options"],
    }


# === Quiz Answer Route ===

@router.post("/submit")
def submit_quiz_answer(
    response: QuizResponse,
    user=Depends(get_current_user),
):
    """
    Validates the user's selected answer.

    Logic:
        - Looks through all quiz entries for one containing the submitted option
        - Checks if it matches the correct answer
        - Returns feedback on whether it was correct or not
    """
    selected_option = response.selected_option.strip()

    if not selected_option:
        raise HTTPException(
            status_code=400,
            detail="No option submitted.",
        )

    for question in quiz_data:
        options = question.get("options", [])

        if selected_option in options:
            answer_index = question.get("answer_index")

            if not isinstance(answer_index, int):
                logger.error(
                    "Invalid answer index in quiz data."
                )
                raise HTTPException(
                    status_code=500,
                    detail="Quiz question is incorrectly configured.",
                )

            try:
                correct_answer = options[answer_index]

            except IndexError:
                logger.error(
                    "Quiz answer index is out of range."
                )
                raise HTTPException(
                    status_code=500,
                    detail="Quiz question is incorrectly configured.",
                ) from None

            is_correct = (
                selected_option == correct_answer
            )

            return {
                "correct_answer": correct_answer,
                "is_correct": is_correct,
            }

    raise HTTPException(
        status_code=400,
        detail="Invalid option submitted.",
    )
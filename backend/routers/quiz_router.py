import json
import logging
import random
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.utils.auth_utils import get_current_user


# quiz_router.py
# ----------------------
# This file handles the routing for the vocabulary quiz section of Fluentique.
# It provides authenticated quiz questions and optional server-side answer
# validation using the bundled curated quiz dataset.


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
    Load and validate quiz questions from the bundled JSON data file.
    """
    try:
        with QUIZ_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:
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

    for index, question in enumerate(data):
        if not isinstance(question, dict):
            raise RuntimeError(
                f"Quiz entry {index} must be an object."
            )

        correct = question.get("correct")
        options = question.get("options")
        answer_index = question.get("answer_index")

        if not isinstance(correct, str) or not correct.strip():
            raise RuntimeError(
                f"Quiz entry {index} has an invalid correct answer."
            )

        if (
            not isinstance(options, list)
            or len(options) < 2
            or not all(
                isinstance(option, str) and option.strip()
                for option in options
            )
        ):
            raise RuntimeError(
                f"Quiz entry {index} has invalid options."
            )

        if (
            not isinstance(answer_index, int)
            or not 0 <= answer_index < len(options)
        ):
            raise RuntimeError(
                f"Quiz entry {index} has an invalid answer_index."
            )

        if options[answer_index] != correct:
            raise RuntimeError(
                f"Quiz entry {index} has inconsistent answer data."
            )

    return data


quiz_data = _load_quiz_data()


# === Request Models ===

class QuizResponse(BaseModel):
    """
    Request body schema for submitting a quiz answer.
    """

    question_id: int
    selected_option: str


# === Quiz Question Route ===

@router.get("/question")
def get_quiz_question(
    user=Depends(get_current_user),
):
    """
    Return a randomly selected quiz question.

    Response:
        - question_id: server-side identifier for answer validation
        - question: correct French word used for TTS playback
        - options: list of possible answers
    """
    question_id = random.randrange(
        len(quiz_data)
    )
    question = quiz_data[
        question_id
    ]

    return {
        "question_id": question_id,
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
    Validate a submitted answer against the exact question that was shown.

    Using a question identifier avoids the ambiguity of searching the entire
    dataset for a matching option, since the same option text may appear in
    more than one quiz question.
    """
    selected_option = (
        response.selected_option.strip()
    )

    if not selected_option:
        raise HTTPException(
            status_code=400,
            detail="No option submitted.",
        )

    if not 0 <= response.question_id < len(quiz_data):
        raise HTTPException(
            status_code=400,
            detail="Invalid question ID.",
        )

    question = quiz_data[
        response.question_id
    ]
    options = question["options"]
    answer_index = question["answer_index"]
    correct_answer = options[
        answer_index
    ]

    if selected_option not in options:
        raise HTTPException(
            status_code=400,
            detail="Selected option does not belong to this question.",
        )

    return {
        "correct_answer": correct_answer,
        "is_correct": selected_option == correct_answer,
    }

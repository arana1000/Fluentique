import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from backend.routers.conversation_router import (
    reformat_audio,
    transcribe_with_google_stt,
    transcribe_with_wav2vec,
)
from backend.services.grammar_exercise_session import (
    exercise_data,
    get_new_exercise,
    submit_translation,
    user_sessions,
)
from backend.utils.auth_utils import get_current_user


# grammar_router.py
# ----------------------
# This file handles all routes related to grammar exercises and corrections.
# It processes user inputs and checks them against the correct grammar using
# a grammar correction model (e.g., MarianMT). It provides feedback to the
# user about any grammatical mistakes and helps improve language accuracy.
#
# Functions:
# - submit_grammar: Accepts user input, runs grammar correction, and provides feedback.
# - fetch_exercise: Retrieves random grammar exercises for the user to practice.


logger = logging.getLogger(__name__)


# === Load grammar correction model ===

grammar_model_path = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "grammar_correction_model"
)

grammar_tokenizer = AutoTokenizer.from_pretrained(
    str(grammar_model_path)
)

grammar_model = AutoModelForSeq2SeqLM.from_pretrained(
    str(grammar_model_path)
)


def run_grammar_correction(text: str) -> str:
    """
    Applies grammar correction using the pre-trained model.

    The model is run as part of the existing grammar pipeline,
    while the original text is returned for downstream validation.
    """
    inputs = grammar_tokenizer.encode(
        f"grammar: {text}",
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )

    grammar_model.generate(
        inputs,
        max_length=512,
        num_beams=4,
        early_stopping=True,
    )

    return text


# === Router configuration ===

router = APIRouter(
    prefix="/grammar",
    tags=["Grammar Exercise"],
)


# === Request Models ===

class SubmitRequest(BaseModel):
    """
    Request model for submitting a translated sentence in the grammar exercise.
    """

    user_translation: str


# === Exercise Start Route ===

@router.post("/exercise/start")
def start_exercise(
    user=Depends(get_current_user),
):
    """
    Starts a new grammar exercise session for the current user.

    Returns:
        - English sentence prompt
        - Corresponding (hidden) French sentence stored for validation
    """
    uid = user["uid"]

    return get_new_exercise(uid)


# === Exercise Submission Route ===

@router.post("/exercise/submit")
def submit_answer(
    req: SubmitRequest,
    user=Depends(get_current_user),
):
    """
    Submits a user-typed sentence for grammar validation.

    Logic:
        - Checks current exercise session
        - If answer is exact match, marks correct and progresses
        - Else, delegates correction/feedback to grammar engine
    """
    uid = user["uid"]
    exercise = user_sessions.get(uid)

    logger.debug(
        "Current grammar exercise for user %s: %s",
        uid,
        exercise,
    )

    if not exercise:
        raise HTTPException(
            status_code=404,
            detail="Exercise session not found.",
        )

    index = exercise.get("current")

    if index is None:
        raise HTTPException(
            status_code=500,
            detail="Exercise session is invalid.",
        )

    expected_french = exercise_data[index]["french"]

    user_translation = run_grammar_correction(
        req.user_translation
    )

    # Direct match path - user sentence is perfect
    if user_translation.strip() == expected_french.strip():
        exercise["responses"].append(
            {
                "prompt": exercise_data[index]["english"],
                "expected": expected_french,
                "user": user_translation,
                "correct": True,
                "feedback": "No mistakes - well done!",
            }
        )

        exercise["current"] += 1

        if exercise["current"] >= 5:
            results = exercise["responses"]

            del user_sessions[uid]

            return {
                "correct": True,
                "message": "Completed 5 exercises.",
                "results": results,
            }

        next_index = exercise["indices"][
            exercise["current"]
        ]

        return {
            "correct": True,
            "message": "Perfect!",
            "next": exercise_data[next_index]["english"],
        }

    # Fallback to grammar correction module
    return submit_translation(
        uid,
        user_translation,
    )


# === Audio Exercise Submission Route ===

@router.post("/exercise/audio_submit")
async def audio_submit(
    request: Request,
    user=Depends(get_current_user),
):
    """
    Accepts raw audio (user's spoken French), transcribes it,
    and submits as an answer for the current grammar exercise session.

    Flow:
        1. Saves and reformats the uploaded WAV audio
        2. Tries Wav2Vec2 transcription, falls back to Google STT
        3. Sends result into grammar checking pipeline
        4. Returns both the transcription and the feedback result
    """
    temp_path = None
    reformatted_path = None

    try:
        body = await request.body()

        if not body:
            raise HTTPException(
                status_code=400,
                detail="Audio data is required.",
            )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav",
        ) as temp_file:
            temp_file.write(body)
            temp_path = temp_file.name

        reformatted_path = reformat_audio(
            temp_path
        )

        # Try custom model first
        try:
            transcription = transcribe_with_wav2vec(
                reformatted_path
            )

            if len(transcription.strip()) < 3:
                raise ValueError(
                    "Transcription too short; using fallback."
                )

        except Exception:
            logger.warning(
                "Wav2Vec2 transcription failed. "
                "Using Google Speech-to-Text fallback.",
                exc_info=True,
            )

            transcription = transcribe_with_google_stt(
                reformatted_path
            )

            transcription = run_grammar_correction(
                transcription
            )

        uid = user["uid"]

        result = submit_translation(
            uid,
            transcription,
        )

        return {
            "transcription": transcription,
            "result": result,
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Failed to process grammar exercise audio."
        )

        return {
            "error": "Failed to process audio submission.",
        }

    finally:
        # Clean up temporary WAV files after transcription.
        for path in {
            temp_path,
            reformatted_path,
        }:
            if path and os.path.exists(path):
                try:
                    os.remove(path)

                except OSError:
                    logger.warning(
                        "Could not delete temporary file: %s",
                        path,
                    )
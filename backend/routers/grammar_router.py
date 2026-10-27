import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from backend.routers.conversation_router import transcribe_with_openai
from backend.services.grammar_exercise_session import (
    get_new_exercise,
    submit_translation,
)
from backend.utils.auth_utils import get_current_user


# grammar_router.py
# ----------------------
# Handles Fluentique's grammar exercise routes.
#
# The grammar exercise service owns session state, answer validation, progression,
# and feedback generation. This router is responsible only for authenticated HTTP
# input/output and audio transcription.
#
# The portfolio/runtime version intentionally avoids local grammar and speech
# models. Spoken answers are transcribed through OpenAI.


logger = logging.getLogger(__name__)


# === Router Configuration ===

router = APIRouter(
    prefix="/grammar",
    tags=["Grammar Exercise"],
)


# === Request Models ===

class SubmitRequest(BaseModel):
    """Request body for a typed French translation."""

    user_translation: str


# === Exercise Start Route ===

@router.post("/exercise/start")
def start_exercise(
    user=Depends(get_current_user),
):
    """
    Start or continue a grammar exercise session for the current user.
    """
    uid = user["uid"]

    try:
        return get_new_exercise(uid)

    except Exception as exc:
        logger.exception(
            "Failed to start grammar exercise for user %s.",
            uid,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to start grammar exercise.",
        ) from exc


# === Typed Exercise Submission Route ===

@router.post("/exercise/submit")
def submit_answer(
    req: SubmitRequest,
    user=Depends(get_current_user),
):
    """
    Submit a typed French translation for validation and feedback.

    Session validation, normalized answer comparison, progression, and optional
    AI feedback are handled by grammar_exercise_session.submit_translation().
    """
    user_translation = req.user_translation.strip()

    if not user_translation:
        raise HTTPException(
            status_code=400,
            detail="Translation is required.",
        )

    uid = user["uid"]

    try:
        return submit_translation(
            uid,
            user_translation,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        logger.exception(
            "Grammar feedback service failed for user %s.",
            uid,
        )
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Failed to submit grammar exercise for user %s.",
            uid,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to submit grammar exercise.",
        ) from exc


# === Audio Exercise Submission Route ===

@router.post("/exercise/audio_submit")
async def audio_submit(
    request: Request,
    user=Depends(get_current_user),
):
    """
    Accept raw WAV audio containing spoken French, transcribe it through OpenAI,
    and submit the resulting text to the same grammar exercise service used by
    typed answers.

    Flow:
        1. Save the uploaded WAV audio to a temporary file.
        2. Transcribe the French speech using OpenAI.
        3. Validate the transcription through the grammar exercise service.
        4. Return both the transcription and grammar result.
    """
    temp_path = None

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

        transcription = await transcribe_with_openai(
            temp_path
        )

        logger.debug(
            "Grammar exercise transcription: %r",
            transcription,
        )

        uid = user["uid"]

        # submit_translation() can call the synchronous OpenAI feedback helper
        # for incorrect answers, so keep it off the async event loop.
        result = await run_in_threadpool(
            submit_translation,
            uid,
            transcription,
        )

        return {
            "transcription": transcription,
            "result": result,
        }

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        logger.exception(
            "Grammar feedback service failed during audio submission."
        )
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Failed to process grammar exercise audio."
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to process audio submission.",
        ) from exc

    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(
                    missing_ok=True
                )

            except OSError:
                logger.warning(
                    "Could not delete temporary file: %s",
                    temp_path,
                )

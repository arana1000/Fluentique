# === IMPORTS AND INITIALISATION ===

# Core audio, API, and utility libraries
import logging
import os
import random
import re
import tempfile
from difflib import SequenceMatcher
from pathlib import Path

import boto3
from pydub import AudioSegment

# FastAPI
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

# OpenAI
from openai import AsyncOpenAI, OpenAIError

# Auth + Logs
from backend.utils.auth_utils import get_current_user
from backend.utils.log_loader import load_logs_summary
from backend.utils.openai_utils import describe_openai_error


# conversation_router.py
# ----------------------
# This file manages Fluentique's conversation-related routes, including:
# - French speech transcription using OpenAI.
# - AI-generated pronunciation/conversation feedback.
# - Free-flowing French tutoring.
# - French-to-English translation.
# - Session summaries.
# - Amazon Polly text-to-speech for generated feedback and tutor replies.
#
# The portfolio/runtime version intentionally avoids loading local speech or
# grammar models here. The original Wav2Vec2/Marian experimentation can be
# preserved separately from the production runtime.


from backend.config import (
    AWS_REGION,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_TRANSCRIPTION_MODEL,
    POLLY_ENGLISH_VOICE_ID,
    POLLY_VOICE_ID,
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"

POLLY_FRENCH_VOICE_ID = POLLY_VOICE_ID

SUPPORTED_AUDIO_SUFFIXES = {
    ".flac",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".m4a",
    ".ogg",
    ".wav",
    ".webm",
}


# AWS Polly client
polly_client = boto3.client(
    "polly",
    region_name=AWS_REGION,
)

# OpenAI client
client_openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

# FastAPI router for conversation and speech endpoints
router = APIRouter(
    prefix="/chat",
    tags=["conversation"],
)


# === General Helpers ===

def _get_audio_suffix(filename: str | None) -> str:
    """
    Return a supported extension for the uploaded audio file.
    Defaults to WAV because Fluentique's current frontend records WAV audio.
    """
    if not filename:
        return ".wav"

    suffix = Path(filename).suffix.lower()

    if suffix in SUPPORTED_AUDIO_SUFFIXES:
        return suffix

    return ".wav"


def _raise_openai_service_error(
    exc: OpenAIError,
    feature: str,
) -> None:
    """Raise a clean HTTP error for expected OpenAI API failures."""
    detail = describe_openai_error(exc)

    # Quota, authentication, rate-limit, and upstream API failures are expected
    # deployment/runtime conditions, so log them without a full traceback.
    logger.warning(
        "%s unavailable: %s",
        feature,
        detail,
    )

    raise HTTPException(
        status_code=503,
        detail=detail,
    ) from exc


async def _create_chat_completion(
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int | None = None,
) -> str:
    """
    Send a chat completion request and return validated response text.
    """
    request_args = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": temperature,
    }

    if max_tokens is not None:
        request_args["max_tokens"] = max_tokens

    response = await client_openai.chat.completions.create(
        **request_args
    )

    content = response.choices[0].message.content

    if not content or not content.strip():
        raise RuntimeError("OpenAI returned an empty response.")

    return content.strip()


# === Audio Helpers ===

def _write_polly_audio_to_temp(
    text: str,
    voice_id: str,
) -> str:
    """
    Generate Polly speech and save it to a temporary MP3 file.
    """
    if not text or not text.strip():
        raise ValueError("Text is required for speech generation.")

    response = polly_client.synthesize_speech(
        VoiceId=voice_id,
        OutputFormat="mp3",
        Text=text,
    )

    audio_stream = response.get("AudioStream")

    if audio_stream is None:
        raise RuntimeError("Amazon Polly returned no audio stream.")

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3",
    ) as temp_file:
        temp_path = temp_file.name

    try:
        with open(temp_path, "wb") as file:
            file.write(audio_stream.read())
    except Exception:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise
    finally:
        audio_stream.close()

    return temp_path


# === Transcription ===

async def transcribe_with_openai(audio_path: str) -> str:
    """
    Transcribe French speech using OpenAI's transcription API.

    The expected sentence is deliberately not supplied as a transcription
    prompt so that recognition is not biased toward what the learner was
    supposed to say.
    """
    with open(audio_path, "rb") as audio_file:
        transcription = await client_openai.audio.transcriptions.create(
            model=OPENAI_TRANSCRIPTION_MODEL,
            file=audio_file,
            language="fr",
        )

    text = getattr(transcription, "text", None)

    if not text:
        raise RuntimeError("OpenAI returned an empty transcription.")

    return text.lower().strip()


def format_transcription(text: str | None) -> str:
    """
    Apply simple formatting to a speech transcription.
    """
    if not text:
        return ""

    text = text.strip().capitalize()

    if not text.endswith("."):
        text += "."

    return text


# === Main Speech Feedback Route ===

@router.post("/transcribe")
async def transcribe_audio(
    request: Request,
    user=Depends(get_current_user),
):
    """
    Accepts uploaded French speech and returns an OpenAI transcription.

    Used by the chatbot voice-input mode so speech recognition remains
    entirely behind the backend API.
    """
    temp_path = None

    try:
        form = await request.form()
        uploaded_file = form.get("file")

        if uploaded_file is None:
            raise HTTPException(
                status_code=400,
                detail="Audio file is required.",
            )

        audio_bytes = await uploaded_file.read()

        if not audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="Audio file is empty.",
            )

        suffix = _get_audio_suffix(
            getattr(uploaded_file, "filename", None)
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        transcription = await transcribe_with_openai(
            temp_path
        )

        return {
            "transcription": format_transcription(
                transcription
            ),
        }

    except HTTPException:
        raise

    except OpenAIError as exc:
        _raise_openai_service_error(
            exc,
            "Speech transcription",
        )

    except RuntimeError as exc:
        logger.exception(
            "OpenAI returned an invalid transcription response."
        )
        raise HTTPException(
            status_code=502,
            detail="Speech transcription returned an invalid response.",
        ) from exc

    except Exception as exc:
        logger.exception(
            "Failed to transcribe uploaded audio."
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to transcribe audio.",
        ) from exc

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning(
                    "Could not delete temporary transcription file: %s",
                    temp_path,
                )

@router.post("/chat_loop")
async def chat_loop(
    request: Request,
    user=Depends(get_current_user),
):
    """
    Accepts spoken audio input and a target sentence.
    Transcribes the learner's French using OpenAI, compares the transcription
    with the expected sentence, and returns teacher feedback plus spoken audio.
    """
    temp_path = None

    try:
        # === Receive audio and target sentence ===
        form = await request.form()

        uploaded_file = form.get("file")
        target_sentence = form.get("target_sentence")

        if uploaded_file is None:
            raise HTTPException(
                status_code=400,
                detail="Audio file is required.",
            )

        if not target_sentence:
            raise HTTPException(
                status_code=400,
                detail="Target sentence is required.",
            )

        audio_bytes = await uploaded_file.read()

        if not audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="Audio file is empty.",
            )

        suffix = _get_audio_suffix(
            getattr(uploaded_file, "filename", None)
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        # === Transcription ===
        transcription = await transcribe_with_openai(temp_path)

        logger.debug(
            "OpenAI transcription: %r",
            transcription,
        )

        transcription = format_transcription(transcription)
        target_sentence = format_transcription(str(target_sentence))

        # === AI Feedback Prompt ===
        prompt = (
            f'The user was supposed to say: "{target_sentence}".\n'
            f'The transcription from their speech is: "{transcription}".\n'
            "Compare the transcription to the target sentence.\n"
            "Ignore differences in capitalisation "
            "(e.g., 'Je' vs 'je').\n"
            "Provide detailed feedback in English (as a teacher) "
            "and pronunciation tips in French (as a native speaker).\n"
            "Only describe pronunciation mistakes that can reasonably be "
            "inferred from the transcription mismatch. Do not pretend that "
            "you directly heard phonetic details that are not represented "
            "in the transcript.\n"
            "Structure your reply like this:\n"
            "Explanation in English: [your explanation]\n"
            "Pronunciation tips in French: [tips in French]\n"
            "Important: Do NOT use IPA symbols "
            "(like /nɑ̃/). Instead, use plain descriptions "
            "like 'noo' for 'nous'."
        )

        reply = await _create_chat_completion(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.7,
            max_tokens=400,
        )

        logger.debug("AI feedback generated successfully.")

        # === Generate Combined Audio ===
        final_mp3_path = await run_in_threadpool(
            generate_combined_audio,
            reply,
        )

        # === Return ===
        return {
            "transcription": transcription,
            "response": reply,
            "audio_file": os.path.basename(final_mp3_path),
        }

    except HTTPException:
        raise

    except OpenAIError as exc:
        _raise_openai_service_error(
            exc,
            "AI speech feedback",
        )

    except RuntimeError as exc:
        logger.exception(
            "OpenAI returned an invalid speech-feedback response."
        )
        raise HTTPException(
            status_code=502,
            detail="AI speech feedback returned an invalid response.",
        ) from exc

    except Exception as exc:
        logger.exception(
            "Failed to process conversation speech."
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to process audio.",
        ) from exc

    finally:
        # Uploaded audio is no longer needed once the request is complete.
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning(
                    "Could not delete temporary file: %s",
                    temp_path,
                )


# === Combined English/French TTS ===

def generate_combined_audio(text: str) -> str:
    """
    Converts the AI feedback (mix of English + French) into a combined MP3.
    Uses AWS Polly, with English text spoken by the English voice and text in
    quotation marks spoken by the French voice.
    Returns the final MP3 path.
    """
    segments = re.split(
        r'(".*?")',
        text,
    )

    combined_audio = None

    for segment in segments:
        if not segment.strip():
            continue

        # Text inside quotation marks is treated as French.
        if segment.startswith('"') and segment.endswith('"'):
            voice_id = POLLY_FRENCH_VOICE_ID
            speech_text = segment.strip('"')
        else:
            voice_id = POLLY_ENGLISH_VOICE_ID
            speech_text = segment

        segment_path = None

        try:
            segment_path = _write_polly_audio_to_temp(
                speech_text,
                voice_id,
            )

            audio_segment = AudioSegment.from_file(segment_path)

            if combined_audio is None:
                combined_audio = audio_segment
            else:
                combined_audio += audio_segment

        finally:
            if segment_path and os.path.exists(segment_path):
                try:
                    os.remove(segment_path)
                except OSError:
                    logger.warning(
                        "Could not delete temporary Polly segment: %s",
                        segment_path,
                    )

    if combined_audio is None:
        raise ValueError(
            "No audio could be generated from the feedback."
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3",
    ) as temp_file:
        final_path = temp_file.name

    try:
        combined_audio.export(
            final_path,
            format="mp3",
        )
    except Exception:
        try:
            os.remove(final_path)
        except OSError:
            pass
        raise

    return final_path


# === Generated Audio Endpoint ===

@router.get("/audio/{filename}")
async def get_audio(filename: str):
    """
    Return a generated TTS MP3 and delete the temporary server-side file
    after the response has finished sending.
    """
    # Only allow normal temporary filenames to prevent directory traversal.
    if not re.fullmatch(
        r"[A-Za-z0-9._-]+",
        filename,
    ):
        raise HTTPException(
            status_code=404,
            detail="Audio file not found.",
        )

    file_path = Path(
        tempfile.gettempdir()
    ) / filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Audio file not found.",
        )

    return FileResponse(
        file_path,
        media_type="audio/mpeg",
        background=BackgroundTask(
            file_path.unlink,
            missing_ok=True,
        ),
    )


# === French Tutor Route ===

@router.post("/tutor")
async def french_tutor(
    request: Request,
    user=Depends(get_current_user),
):
    """
    Handles free-flowing chat messages from the user.
    Uses previous logs to revisit grammar or pronunciation mistakes if relevant.
    Gives personalised replies mixing French and English using OpenAI.
    Returns both text and a spoken version (MP3).
    """

    def similarity(a: str, b: str) -> float:
        return SequenceMatcher(
            None,
            a.lower(),
            b.lower(),
        ).ratio()

    body = await request.json()

    user_input = body.get(
        "message",
        "",
    ).strip()

    is_first = body.get(
        "is_first",
        False,
    )

    if not is_first and not user_input:
        raise HTTPException(
            status_code=400,
            detail="Message is required.",
        )

    base_prompt = (
        "Tu es un tuteur de français bienveillant, patient et professionnel. "
        "Tu parles principalement en français, sauf pour quelques "
        "encouragements ou explications claires en anglais. "
        "Utilise un langage simple (niveau A2). "
        "Très important : ne corrige que si la phrase contient une erreur claire. "
        "Si la phrase est correcte, dis-le, et propose une variante ou une autre "
        "phrase liée, sans répéter la même chose. "
        "Évite d'inventer des fautes imaginaires ou de corriger les phrases "
        "déjà correctes. "
        "Fais progresser la conversation naturellement. "
        "Keep replies as short and concise as possible whilst also being "
        "friendly and teaching. Try switching between French and English "
        "when you think is best."
    )

    messages = [
        {
            "role": "system",
            "content": base_prompt,
        }
    ]

    try:
        # Greet user cleanly and exit early on first entry.
        if is_first:
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "Bonjour ! Comment tu vas aujourd’hui ? "
                        "On va revoir ensemble quelques phrases que tu as eues "
                        "du mal à dire ou écrire. "
                        "Mais d’abord, dis-moi quelque chose - n’importe quoi - "
                        "pour commencer la discussion."
                    ),
                }
            )

            reply = await _create_chat_completion(
                messages,
                temperature=0.7,
            )

            audio_path = await run_in_threadpool(
                _write_polly_audio_to_temp,
                reply,
                POLLY_FRENCH_VOICE_ID,
            )

            return {
                "reply": reply,
                "audio_file": os.path.basename(audio_path),
            }

        # === For all non-first messages, use logs for corrections ===
        try:
            logs = load_logs_summary()
        except Exception:
            # Tutor chat should still work even if historical logs are malformed
            # or temporarily unavailable.
            logger.exception(
                "Failed to load tutor history. Continuing without log context."
            )
            logs = {
                "mistakes": [],
                "speech_issues": [],
            }

        mistakes = logs.get("mistakes", [])[-5:]
        speech_issues = logs.get("speech_issues", [])[-5:]

        used_phrases = set()
        correction_added = False
        all_issues = []

        for user_mistake, expected in mistakes:
            all_issues.append(
                (
                    "text",
                    user_mistake,
                    expected,
                    None,
                )
            )

        for target, said, feedback in speech_issues:
            all_issues.append(
                (
                    "speech",
                    said,
                    target,
                    feedback,
                )
            )

        random.shuffle(all_issues)

        # Lowered threshold to increase match chance.
        for (
            issue_type,
            user_phrase,
            expected,
            feedback,
        ) in all_issues:
            if (
                similarity(
                    user_input,
                    user_phrase,
                )
                > 0.75
                and expected.lower() not in used_phrases
            ):
                if issue_type == "text":
                    messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                f"Presque ! La bonne phrase est : '{expected}'. "
                                f"Répète après moi : {expected}. "
                                f"Encore une fois : {expected}. "
                                "Peux-tu faire une phrase avec cela ?"
                            ),
                        }
                    )

                else:
                    feedback_line = (
                        feedback.splitlines()[0]
                        if feedback
                        else ""
                    )

                    messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                f"Tu avais dit : '{user_phrase}' "
                                f"au lieu de '{expected}'. "
                                f"{feedback_line} "
                                f"Répète après moi : {expected}. "
                                f"Encore une fois : {expected}. "
                                "Et toi ? Tu peux l'utiliser dans une phrase ?"
                            ),
                        }
                    )

                used_phrases.add(expected.lower())
                correction_added = True
                break

        # Force log injection with 25% chance if no match.
        if (
            not correction_added
            and random.random() < 0.25
            and all_issues
        ):
            (
                issue_type,
                user_phrase,
                expected,
                feedback,
            ) = random.choice(all_issues)

            if issue_type == "text":
                messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "Rappel rapide ! Tu avais écrit : "
                            f"'{user_phrase}'. "
                            f"Voici la bonne version : '{expected}'. "
                            "Peux-tu la redire ?"
                        ),
                    }
                )

            else:
                feedback_line = (
                    feedback.splitlines()[0]
                    if feedback
                    else ""
                )

                messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "Souviens-toi : tu avais prononcé "
                            f"'{user_phrase}' au lieu de '{expected}'. "
                            f"{feedback_line} "
                            f"Répète : {expected}"
                        ),
                    }
                )

            correction_added = True

        if correction_added:
            messages.append(
                {
                    "role": "user",
                    "content": user_input,
                }
            )

        else:
            practice = random.choice(
                [
                    "Dis-moi quelque chose que tu as appris aujourd'hui.",
                    "Essaie de poser une question simple en français.",
                    (
                        "Voici une nouvelle phrase utile : "
                        '"Je ferme la fenêtre." '
                        "Peux-tu la répéter ?"
                    ),
                ]
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        random.choice(
                            [
                                "Très bien ! Continuons.",
                                "Parfait. Allons-y.",
                            ]
                        )
                        + " "
                        + practice
                    ),
                }
            )

            messages.append(
                {
                    "role": "user",
                    "content": user_input,
                }
            )

        reply = await _create_chat_completion(
            messages,
            temperature=0.7,
        )

        audio_path = await run_in_threadpool(
            _write_polly_audio_to_temp,
            reply,
            POLLY_FRENCH_VOICE_ID,
        )

        return {
            "reply": reply,
            "audio_file": os.path.basename(audio_path),
        }

    except HTTPException:
        raise

    except OpenAIError as exc:
        _raise_openai_service_error(
            exc,
            "AI tutor",
        )

    except RuntimeError as exc:
        logger.exception("OpenAI returned an invalid tutor response.")
        raise HTTPException(
            status_code=502,
            detail="AI tutor returned an invalid response.",
        ) from exc

    except Exception as exc:
        logger.exception("French tutor request failed.")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate tutor response.",
        ) from exc


# === Translation Route ===

@router.post("/translate")
async def translate_text(
    request: Request,
    user=Depends(get_current_user),
):
    """
    Translates a French sentence to English using OpenAI.
    Meant for clarifying AI replies or unknown words.
    """
    body = await request.json()
    text = body.get("text", "").strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text is required.",
        )

    system_prompt = (
        "Translate this French message to English. "
        "Be natural and accurate. "
        "Do not include any extra explanation."
    )

    try:
        english = await _create_chat_completion(
            [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            temperature=0.3,
        )

        return {
            "translation": english,
        }

    except OpenAIError as exc:
        _raise_openai_service_error(
            exc,
            "AI translation",
        )

    except RuntimeError as exc:
        logger.exception("OpenAI returned an invalid translation response.")
        raise HTTPException(
            status_code=502,
            detail="AI translation returned an invalid response.",
        ) from exc

    except Exception as exc:
        logger.exception("Translation request failed.")
        raise HTTPException(
            status_code=500,
            detail="Failed to translate text.",
        ) from exc


# === Session Summary Route ===

@router.post("/summary")
async def generate_session_summary(
    user=Depends(get_current_user),
):
    """
    Analyses the latest chat transcript using OpenAI to produce a short,
    friendly performance summary mentioning strengths and improvement areas.
    """
    # Directory where logs are stored.
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Find the most recent chat log.
    log_files = sorted(
        LOG_DIR.glob("chat_transcript_*.txt"),
        key=os.path.getmtime,
        reverse=True,
    )

    if not log_files:
        return {
            "summary": "No transcript found to generate a summary."
        }

    recent_log = log_files[0]

    try:
        transcript_text = recent_log.read_text(
            encoding="utf-8"
        )

    except OSError as exc:
        logger.exception(
            "Failed to read chat transcript: %s",
            recent_log,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to read session transcript.",
        ) from exc

    # Use OpenAI to summarise the conversation.
    try:
        summary_prompt = (
            "You are a kind, patient, and professional French language tutor. "
            "Read the following transcript of a tutoring session between you "
            "and a student. "
            "Write a short summary in **English** that gives feedback on their "
            "performance today. "
            "Be warm and encouraging. Focus on their **strengths**, and mention "
            "**one or two areas they could improve**. "
            "Keep the summary concise - no more than **80 words**. "
            "Do not repeat transcript lines or sound robotic.\n\n"
            f"{transcript_text}"
        )

        summary = await _create_chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a French learning tutor "
                        "summarising student progress in English."
                    ),
                },
                {
                    "role": "user",
                    "content": summary_prompt,
                },
            ],
            temperature=0.6,
        )

        return {
            "summary": summary,
        }

    except OpenAIError as exc:
        _raise_openai_service_error(
            exc,
            "AI session summary",
        )

    except RuntimeError as exc:
        logger.exception(
            "OpenAI returned an invalid session-summary response."
        )
        raise HTTPException(
            status_code=502,
            detail="AI session summary returned an invalid response.",
        ) from exc

    except Exception:
        logger.exception(
            "Session summarisation failed."
        )

        return {
            "summary": "Failed to generate session summary.",
        }
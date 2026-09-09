# === IMPORTS AND INITIALISATION ===

# Core speech, audio, ML, and API libraries
import io
import logging
import os
import random
import re
import tempfile
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path

import boto3
import google.cloud.speech as speech
import librosa
import soundfile as sf
import torch
from pydub import AudioSegment

# FastAPI
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

# Transformers + OpenAI
from openai import OpenAI, OpenAIError
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

# Auth + Logs
from backend.config import GOOGLE_CREDS_PATH, OPENAI_API_KEY
from backend.utils.auth_utils import get_current_user
from backend.utils.log_loader import load_logs_summary


# conversation_router.py
# ----------------------
# This file manages the conversation-related routes for the app, focusing on
# speech-to-text conversion and real-time feedback generation.
# It uses Wav2Vec2 and Google Speech-to-Text for speech recognition and the
# Grammar Model results passed to GPT to generate feedback based on the
# transcribed text. This allows the app to provide real-time conversational
# practice for users.
#
# Functions:
# - transcribe_speech: Converts spoken input into text using Wav2Vec2 or
#   Google Speech-to-Text.
# - provide_feedback: Generates feedback for the transcribed text using
#   Grammar model results passed to GPT.


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
AWS_REGION = os.getenv("AWS_REGION", "eu-west-2")


# Conversation memory values retained for compatibility with existing logic
chat_memory = deque(maxlen=6)
last_user_sentence = ""
repeat_count = 0


# AWS Polly client
polly_client = boto3.client(
    "polly",
    region_name=AWS_REGION,
)

# FastAPI router for conversation and speech endpoints
router = APIRouter(
    prefix="/chat",
    tags=["conversation"],
)


# === Load custom Wav2Vec2 model ===

MODEL_DIR = BASE_DIR / "models" / "fluentique-wav2vec2-fr"
MODEL_PATH = str(MODEL_DIR)

processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)


# === Google STT setup ===

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(GOOGLE_CREDS_PATH)
client_google = speech.SpeechClient()


# === OpenAI GPT ===

client_openai = OpenAI(api_key=OPENAI_API_KEY)


# === Load grammar correction model ===

GRAMMAR_MODEL_DIR = BASE_DIR / "models" / "grammar_correction_model"

grammar_tokenizer = AutoTokenizer.from_pretrained(
    str(GRAMMAR_MODEL_DIR)
)

grammar_model = AutoModelForSeq2SeqLM.from_pretrained(
    str(GRAMMAR_MODEL_DIR)
)


# === Audio Helpers ===

def reformat_audio(input_path: str) -> str:
    """
    Rewrites an audio file as PCM 16-bit WAV for speech recognition.
    """
    data, samplerate = sf.read(input_path)

    logger.debug(
        "Audio shape: %s, samplerate: %s",
        data.shape,
        samplerate,
    )

    input_file = Path(input_path)
    reformatted_path = input_file.with_name(
        f"{input_file.stem}_reformatted.wav"
    )

    sf.write(
        reformatted_path,
        data,
        samplerate,
        subtype="PCM_16",
    )

    return str(reformatted_path)


def _write_polly_audio_to_temp(
    text: str,
    voice_id: str,
) -> str:
    """
    Generate Polly speech and save it to a temporary MP3 file.
    """
    response = polly_client.synthesize_speech(
        VoiceId=voice_id,
        OutputFormat="mp3",
        Text=text,
    )

    audio_stream = response.get("AudioStream")

    if audio_stream is None:
        raise RuntimeError("Amazon Polly returned no audio stream.")

    temp_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3",
    ).name

    try:
        with open(temp_path, "wb") as file:
            file.write(audio_stream.read())
    finally:
        audio_stream.close()

    return temp_path


# === Transcription functions ===

def transcribe_with_wav2vec(audio_path: str) -> str:
    """
    Transcribe French speech using the custom Wav2Vec2 model.
    """
    speech_array, _ = librosa.load(
        audio_path,
        sr=16000,
    )

    inputs = processor(
        speech_array,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    pred_ids = torch.argmax(
        logits,
        dim=-1,
    )

    return (
        processor.batch_decode(pred_ids)[0]
        .lower()
        .strip()
    )


def transcribe_with_google_stt(audio_path: str) -> str:
    """
    Transcribe French speech using Google Speech-to-Text.
    """
    with io.open(audio_path, "rb") as audio_file:
        content = audio_file.read()

    audio_info = sf.info(audio_path)

    audio = speech.RecognitionAudio(
        content=content,
    )

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=audio_info.samplerate,
        language_code="fr-FR",
    )

    response = client_google.recognize(
        config=config,
        audio=audio,
    )

    if response.results:
        return (
            response.results[0]
            .alternatives[0]
            .transcript
            .lower()
            .strip()
        )

    return ""


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


# === Grammar Correction ===

def run_grammar_correction(text: str) -> str:
    """
    Applies grammar correction using the pre-trained model.
    Logs the original and corrected version, but returns the original.
    """
    inputs = grammar_tokenizer.encode(
        f"grammar: {text}",
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )

    outputs = grammar_model.generate(
        inputs,
        max_length=512,
        num_beams=4,
        early_stopping=True,
    )

    corrected = grammar_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    ).strip()

    # Log if there's a visible difference
    if corrected.lower() != text.lower():
        LOG_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        log_path = LOG_DIR / "grammar_corrections.txt"

        with log_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                f"Original: {text}\n"
                f"Corrected: {corrected}\n\n"
            )

    return text


# === Main Speech Feedback Route ===

@router.post("/chat_loop")
async def chat_loop(request: Request):
    """
    Accepts a spoken audio input and target sentence.
    Transcribes it using Wav2Vec2 (with Google STT fallback).
    Uses GPT to compare the expected vs actual speech and provide feedback.
    Returns feedback text and an AI-generated audio file that reads it aloud.
    """
    temp_path = None
    reformatted_path = None

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

        audio_file = await uploaded_file.read()

        if not audio_file:
            raise HTTPException(
                status_code=400,
                detail="Audio file is empty.",
            )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav",
        ) as temp_file:
            temp_file.write(audio_file)
            temp_path = temp_file.name

        reformatted_path = reformat_audio(temp_path)

        logger.debug(
            "Reformatted audio path: %s",
            reformatted_path,
        )

        # === Transcription ===
        # Wav2Vec2 is attempted first, with Google STT used as the fallback.
        try:
            transcription = transcribe_with_wav2vec(
                reformatted_path
            )

            logger.debug(
                "Wav2Vec2 transcription: %r",
                transcription,
            )

            if len(transcription.strip()) < 3:
                raise ValueError(
                    "Wav2Vec2 transcription was too short."
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

            logger.debug(
                "Google STT fallback output: %r",
                transcription,
            )

        transcription = format_transcription(
            transcription
        )

        target_sentence = format_transcription(
            str(target_sentence)
        )

        # === GPT Feedback Prompt ===
        prompt = (
            f'The user was supposed to say: "{target_sentence}".\n'
            f'The transcription from their speech is: "{transcription}".\n'
            "Compare the transcription to the target sentence.\n"
            "Ignore differences in capitalisation "
            "(e.g., 'Je' vs 'je').\n"
            "Provide detailed feedback in English (as a teacher) "
            "and pronunciation tips in French (as a native speaker).\n"
            "Structure your reply like this:\n"
            "Explanation in English: [your explanation]\n"
            "Pronunciation tips in French: [tips in French]\n"
            "Important: Do NOT use IPA symbols "
            "(like /nɑ̃/). Instead, use plain descriptions "
            "like 'noo' for 'nous'."
        )

        response = client_openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.7,
            max_tokens=400,
        )

        reply = (
            response.choices[0]
            .message.content
            .strip()
        )

        logger.debug("GPT feedback generated successfully.")

        # === Generate Combined Audio ===
        final_mp3_path = generate_combined_audio(
            reply
        )

        # === Return ===
        return {
            "transcription": transcription,
            "response": reply,
            "audio_file": os.path.basename(
                final_mp3_path
            ),
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Failed to process conversation speech."
        )

        return {
            "error": "Failed to process audio.",
        }

    finally:
        # Temporary WAV files are no longer needed after processing.
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


# === Combined English/French TTS ===

def generate_combined_audio(text: str) -> str:
    """
    Converts the GPT feedback (mix of English + French) into a combined MP3.
    Uses AWS Polly - English text spoken by 'Amy', French by 'Celine'.
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
        if (
            segment.startswith('"')
            and segment.endswith('"')
        ):
            voice_id = "Celine"
            speech_text = segment.strip('"')
        else:
            voice_id = "Amy"
            speech_text = segment

        segment_path = None

        try:
            segment_path = _write_polly_audio_to_temp(
                speech_text,
                voice_id,
            )

            audio_segment = AudioSegment.from_file(
                segment_path
            )

            if combined_audio is None:
                combined_audio = audio_segment
            else:
                combined_audio += audio_segment

        finally:
            if (
                segment_path
                and os.path.exists(segment_path)
            ):
                try:
                    os.remove(segment_path)
                except OSError:
                    logger.warning(
                        "Could not delete temporary "
                        "Polly segment: %s",
                        segment_path,
                    )

    if combined_audio is None:
        raise ValueError(
            "No audio could be generated from the feedback."
        )

    final_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3",
    ).name

    combined_audio.export(
        final_path,
        format="mp3",
    )

    return final_path


# === Generated Audio Endpoint ===

@router.get("/audio/{filename}")
async def get_audio(filename: str):
    """
    Endpoint to return TTS MP3 files generated during pronunciation feedback.
    """
    # Only allow normal temporary filenames to prevent directory traversal.
    if not re.fullmatch(
        r"[A-Za-z0-9._-]+",
        filename,
    ):
        return {"error": "File not found"}

    file_path = (
        Path(tempfile.gettempdir())
        / filename
    )

    if file_path.exists() and file_path.is_file():
        return FileResponse(
            file_path,
            media_type="audio/mpeg",
        )

    return {"error": "File not found"}


# === French Tutor Route ===

@router.post("/tutor")
async def french_tutor(
    request: Request,
    user=Depends(get_current_user),
):
    """
    Handles free-flowing chat messages from the user.
    Uses previous logs to correct grammar or pronunciation mistakes if relevant.
    Gives personalised replies mixing French and English using GPT.
    Returns both text and spoken version (MP3).
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

    user_input = run_grammar_correction(
        user_input
    )

    is_first = body.get(
        "is_first",
        False,
    )

    # === For all non-first messages, confirm the pronunciation model is loaded ===
    if not is_first:
        logger.debug(
            "(Tutor) Pronunciation model is loaded "
            "and available."
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

    # Greet user cleanly and exit early on first entry
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

        response = client_openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.7,
        )

        reply = (
            response.choices[0]
            .message.content
            .strip()
        )

        audio_path = _write_polly_audio_to_temp(
            reply,
            "Celine",
        )

        return {
            "reply": reply,
            "audio_file": os.path.basename(
                audio_path
            ),
        }

    # === For all non-first messages, use logs for corrections ===
    logs = load_logs_summary()

    mistakes = logs["mistakes"][-5:]
    speech_issues = logs["speech_issues"][-5:]

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

    # Lowered threshold to increase match chance
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
            and expected.lower()
            not in used_phrases
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

            used_phrases.add(
                expected.lower()
            )

            correction_added = True
            break

    # Force log injection with 25% chance if no match
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
                        f"Rappel rapide ! Tu avais écrit : "
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
                        f"Souviens-toi : tu avais prononcé "
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
                    'Voici une nouvelle phrase utile : '
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

    # Call GPT
    response = client_openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.7,
    )

    reply = (
        response.choices[0]
        .message.content
        .strip()
    )

    audio_path = _write_polly_audio_to_temp(
        reply,
        "Celine",
    )

    return {
        "reply": reply,
        "audio_file": os.path.basename(
            audio_path
        ),
    }


# === Translation Route ===

@router.post("/translate")
async def translate_text(
    request: Request,
    user=Depends(get_current_user),
):
    """
    Translates a French sentence to English using GPT.
    Meant for clarifying AI replies or unknown words.
    """
    body = await request.json()
    text = body.get("text", "")

    system_prompt = (
        "Translate this French message to English. "
        "Be natural and accurate. "
        "Do not include any extra explanation."
    )

    response = client_openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
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

    english = (
        response.choices[0]
        .message.content
        .strip()
    )

    return {
        "translation": english,
    }


# === Session Summary Route ===

@router.post("/summary")
async def generate_session_summary(
    user=Depends(get_current_user),
):
    """
    Analyses the latest chat transcript using GPT to produce a short, friendly
    performance summary mentioning strengths and improvement areas.
    """
    # Directory where logs are stored
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Find the most recent chat log
    log_files = sorted(
        LOG_DIR.glob(
            "chat_transcript_*.txt"
        ),
        key=os.path.getmtime,
        reverse=True,
    )

    if not log_files:
        return {
            "summary": (
                "No transcript found to generate a summary."
            )
        }

    recent_log = log_files[0]

    try:
        transcript_text = recent_log.read_text(
            encoding="utf-8"
        )

    except OSError:
        logger.exception(
            "Failed to read chat transcript: %s",
            recent_log,
        )

        return {
            "summary": "Failed to read log file.",
        }

    # Use OpenAI to summarise the conversation
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

        response = client_openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
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

        summary = (
            response.choices[0]
            .message.content
            .strip()
        )

        return {
            "summary": summary,
        }

    except OpenAIError:
        logger.exception(
            "OpenAI session summarisation failed."
        )

        return {
            "summary": "OpenAI summarisation failed.",
        }
import logging
import threading
from xml.sax.saxutils import escape

import boto3
import numpy as np
import sounddevice as sd

from backend.config import (
    AWS_REGION,
    POLLY_LANGUAGE_CODE,
    POLLY_VOICE_ID,
)


# tts_utils.py
# ----------------------
# Shared Amazon Polly text-to-speech helper used by Fluentique.
#
# AWS credentials are intentionally not stored in the project. Boto3 uses its
# normal credential chain, while region, voice, and language settings come
# from backend.config.
#
# Polly returns raw PCM audio here, which is played through SoundDevice.
# Short vocabulary words are spoken slightly more slowly and padded with
# silence so Windows audio-device startup does not clip the first or last
# phoneme.


logger = logging.getLogger(__name__)


# === Amazon Polly Client ===

polly = boto3.client(
    "polly",
    region_name=AWS_REGION,
)

POLLY_SAMPLE_RATE = 16000
POLLY_SPEECH_RATE = "85%"
LEAD_IN_MS = 250
TAIL_MS = 400

# Prevent overlapping replay requests from fighting over the audio device.
_playback_lock = threading.Lock()


def _silence(duration_ms: int) -> np.ndarray:
    """
    Return signed 16-bit mono silence for the configured Polly sample rate.
    """
    sample_count = int(
        POLLY_SAMPLE_RATE * duration_ms / 1000
    )

    return np.zeros(
        sample_count,
        dtype=np.int16,
    )


def play_word_with_polly(word: str) -> bool:
    """
    Synthesise and play a French word or short phrase using Amazon Polly.

    Short vocabulary items are deliberately spoken at 85% speed. A short
    silence buffer is also added before and after the generated PCM audio to
    prevent audio-device startup/shutdown from clipping word boundaries.

    Args:
        word (str): French text to speak.

    Returns:
        bool: True if playback completed successfully, otherwise False.
    """
    text = word.strip()

    if not text:
        logger.warning(
            "Polly playback requested with empty text."
        )
        return False

    if not _playback_lock.acquire(blocking=False):
        logger.debug(
            "Ignoring Polly replay while audio is already playing."
        )
        return False

    audio_stream = None

    try:
        ssml_text = (
            "<speak>"
            f"<prosody rate=\"{POLLY_SPEECH_RATE}\">"
            f"{escape(text)}"
            "</prosody>"
            "</speak>"
        )

        response = polly.synthesize_speech(
            Text=ssml_text,
            TextType="ssml",
            OutputFormat="pcm",
            SampleRate=str(POLLY_SAMPLE_RATE),
            VoiceId=POLLY_VOICE_ID,
            LanguageCode=POLLY_LANGUAGE_CODE,
        )

        audio_stream = response.get(
            "AudioStream"
        )

        if audio_stream is None:
            raise RuntimeError(
                "Amazon Polly returned no audio stream."
            )

        audio_bytes = audio_stream.read()

        if not audio_bytes:
            raise RuntimeError(
                "Amazon Polly returned empty audio."
            )

        speech_samples = np.frombuffer(
            audio_bytes,
            dtype=np.int16,
        )

        playback_samples = np.concatenate(
            (
                _silence(LEAD_IN_MS),
                speech_samples,
                _silence(TAIL_MS),
            )
        )

        sd.play(
            playback_samples,
            samplerate=POLLY_SAMPLE_RATE,
            blocking=True,
        )

        return True

    except Exception:
        logger.exception(
            "Amazon Polly playback failed for text: %r",
            text,
        )

        try:
            sd.stop()
        except Exception:
            pass

        return False

    finally:
        if audio_stream is not None:
            try:
                audio_stream.close()
            except Exception:
                logger.debug(
                    "Failed to close Polly audio stream.",
                    exc_info=True,
                )

        _playback_lock.release()

import logging
from pathlib import Path


# log_loader.py
# ----------------------
# Loads Fluentique's local learning-history logs into a compact structure
# that can be reused by the AI tutor for personalised revision.


logger = logging.getLogger(__name__)

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

SENTENCE_LOG = LOG_DIR / "sentence_exercise_log.txt"
SPEECH_LOG = LOG_DIR / "speech_feedback_log.txt"


def load_logs_summary() -> dict[str, list[tuple[str, ...]]]:
    """
    Load a summarised view of sentence and speech feedback logs.

    Returns:
        dict: Contains:
            - mistakes: (user_input, expected_output)
            - speech_issues: (target_sentence, user_input, feedback)
    """
    summary: dict[str, list[tuple[str, ...]]] = {
        "mistakes": [],
        "speech_issues": [],
    }

    _load_sentence_mistakes(summary)
    _load_speech_issues(summary)

    return summary


def _load_sentence_mistakes(
    summary: dict[str, list[tuple[str, ...]]],
) -> None:
    """
    Extract incorrect grammar attempts and their expected answers.
    """
    if not SENTENCE_LOG.exists():
        return

    try:
        lines = SENTENCE_LOG.read_text(
            encoding="utf-8"
        ).splitlines()

        pending_user_input: str | None = None

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("❌ User:"):
                pending_user_input = stripped.removeprefix(
                    "❌ User:"
                ).strip()
                continue

            if (
                pending_user_input
                and stripped.startswith("✅ Expected:")
            ):
                expected_output = stripped.removeprefix(
                    "✅ Expected:"
                ).strip()

                if expected_output:
                    summary["mistakes"].append(
                        (
                            pending_user_input,
                            expected_output,
                        )
                    )

                pending_user_input = None

    except OSError:
        logger.exception(
            "Failed to read sentence exercise log: %s",
            SENTENCE_LOG,
        )


def _load_speech_issues(
    summary: dict[str, list[tuple[str, ...]]],
) -> None:
    """
    Extract speech-practice targets, transcriptions, and feedback.

    Entries are detected by their log prefixes rather than fixed line
    positions so a malformed or extra line does not break later entries.
    """
    if not SPEECH_LOG.exists():
        return

    try:
        lines = SPEECH_LOG.read_text(
            encoding="utf-8"
        ).splitlines()

        current_sentence = ""
        current_user_input = ""

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("🎯 Sentence:"):
                current_sentence = stripped.removeprefix(
                    "🎯 Sentence:"
                ).strip()
                current_user_input = ""
                continue

            if stripped.startswith("🗣️ User Said:"):
                current_user_input = stripped.removeprefix(
                    "🗣️ User Said:"
                ).strip()
                continue

            if stripped.startswith("💬 Feedback:"):
                feedback = stripped.removeprefix(
                    "💬 Feedback:"
                ).strip()

                if (
                    current_sentence
                    and current_user_input
                    and feedback
                ):
                    summary["speech_issues"].append(
                        (
                            current_sentence,
                            current_user_input,
                            feedback,
                        )
                    )

                current_sentence = ""
                current_user_input = ""

    except OSError:
        logger.exception(
            "Failed to read speech feedback log: %s",
            SPEECH_LOG,
        )

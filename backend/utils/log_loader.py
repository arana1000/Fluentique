import logging
from pathlib import Path


logger = logging.getLogger(__name__)

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

SENTENCE_LOG = LOG_DIR / "sentence_exercise_log.txt"
SPEECH_LOG = LOG_DIR / "speech_feedback_log.txt"


def load_logs_summary() -> dict[str, list[tuple[str, ...]]]:
    """Load a summarised view of sentence and speech feedback logs."""
    summary = {
        "mistakes": [],
        "speech_issues": [],
    }

    _load_sentence_mistakes(summary)
    _load_speech_issues(summary)

    return summary


def _load_sentence_mistakes(
    summary: dict[str, list[tuple[str, ...]]],
) -> None:
    if not SENTENCE_LOG.exists():
        return

    try:
        lines = SENTENCE_LOG.read_text(encoding="utf-8").splitlines()

        for index, line in enumerate(lines):
            if not line.startswith("❌"):
                continue

            user_input = line.replace("❌ User:", "", 1).strip()

            expected_output = ""
            if index + 1 < len(lines):
                expected_output = lines[index + 1].replace(
                    "✅ Expected:",
                    "",
                    1,
                ).strip()

            if user_input and expected_output:
                summary["mistakes"].append(
                    (user_input, expected_output)
                )

    except OSError:
        logger.exception(
            "Failed to read sentence exercise log: %s",
            SENTENCE_LOG,
        )


def _load_speech_issues(
    summary: dict[str, list[tuple[str, ...]]],
) -> None:
    if not SPEECH_LOG.exists():
        return

    try:
        lines = SPEECH_LOG.read_text(encoding="utf-8").splitlines()

        for index in range(0, len(lines), 4):
            if index + 2 >= len(lines):
                break

            sentence = lines[index].split(
                "🎯 Sentence:",
                1,
            )[-1].strip()

            user_input = lines[index + 1].split(
                "🗣️ User Said:",
                1,
            )[-1].strip()

            feedback = lines[index + 2].split(
                "💬 Feedback:",
                1,
            )[-1].strip()

            if sentence and user_input and feedback:
                summary["speech_issues"].append(
                    (sentence, user_input, feedback)
                )

    except OSError:
        logger.exception(
            "Failed to read speech feedback log: %s",
            SPEECH_LOG,
        )

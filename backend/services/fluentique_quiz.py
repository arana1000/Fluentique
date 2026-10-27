# === Imports ===

import csv
import json
import logging
import random
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from backend.config import AWS_REGION
from backend.utils.tts_utils import play_word_with_polly


# fluentique_quiz.py
# ----------------------
# This file provides shared vocabulary-quiz helpers used by the Kivy frontend
# and retains the original command-line listening quiz for manual use.
#
# The portfolio/runtime version no longer loads the experimental pickled
# difficulty-prediction model. The CLI still adapts difficulty using learner
# streaks, and its challenge round prioritises hard words the learner missed
# during the current session.
#
# Functions:
# - get_word_info: Translates a French word and provides a simple definition.
# - translate_word: Returns an English translation for quiz feedback.
# - run_cli_quiz: Runs the standalone adaptive listening quiz.


logger = logging.getLogger(__name__)

translate_client = boto3.client(
    "translate",
    region_name=AWS_REGION,
)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "french_quiz_data_aws.json"
LOGS_DIR = BASE_DIR / "logs"

MAIN_QUIZ_LENGTH = 20
CHALLENGE_LENGTH = 5

LEVELS = (
    "easy",
    "moderate",
    "hard",
)


# === Load static French quiz dataset ===

def _load_quiz_data() -> list[dict]:
    """
    Load and validate the curated French listening-quiz dataset.
    """
    try:
        with DATA_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)

    except (OSError, json.JSONDecodeError) as exc:
        logger.exception(
            "Failed to load quiz data from %s.",
            DATA_PATH,
        )

        raise RuntimeError(
            f"Unable to load quiz data from {DATA_PATH}."
        ) from exc

    if not isinstance(data, list) or not data:
        raise RuntimeError(
            "Quiz data must contain a non-empty list of questions."
        )

    for index, question in enumerate(data):
        if not isinstance(question, dict):
            raise RuntimeError(
                f"Quiz question at index {index} is invalid."
            )

        correct = question.get("correct")
        options = question.get("options")
        answer_index = question.get("answer_index")
        level = question.get("level")

        if (
            not isinstance(correct, str)
            or not correct.strip()
            or not isinstance(options, list)
            or not options
            or not all(
                isinstance(option, str) and option.strip()
                for option in options
            )
            or not isinstance(answer_index, int)
            or not 0 <= answer_index < len(options)
            or options[answer_index] != correct
            or level not in LEVELS
        ):
            raise RuntimeError(
                f"Quiz question at index {index} is incorrectly configured."
            )

    return data


quiz_data = _load_quiz_data()


# === Organise questions by difficulty level ===

by_level = {
    level: [
        question
        for question in quiz_data
        if question["level"] == level
    ]
    for level in LEVELS
}


# === Translation + definition utilities for feedback ===

@lru_cache(maxsize=256)
def translate_word(word: str) -> str:
    """
    Translate a French word to English for quiz feedback using Amazon Translate.

    Successful lookups are cached for the current process so repeated quiz
    feedback does not make unnecessary cloud requests.
    """
    text = word.strip()

    if not text:
        return "Translation unavailable"

    try:
        response = translate_client.translate_text(
            Text=text,
            SourceLanguageCode="fr",
            TargetLanguageCode="en",
        )

        translated = response.get(
            "TranslatedText",
            "",
        ).strip()

        if not translated:
            raise RuntimeError(
                "Amazon Translate returned an empty translation."
            )

        return translated

    except (BotoCoreError, ClientError, RuntimeError) as exc:
        logger.warning(
            "Unable to translate French quiz word %r: %s",
            text,
            exc,
        )

        return "Translation unavailable"

def get_word_info(word: str) -> dict[str, str]:
    """
    Translate a French word and provide a simple learner-friendly definition.

    This helper is currently imported by the Kivy frontend.
    """
    text = word.strip()

    if not text:
        return {
            "translation": "Unavailable",
            "definition": "Definition unavailable",
        }

    translated = translate_word(text)

    if translated == "Translation unavailable":
        return {
            "translation": "Unavailable",
            "definition": "Definition unavailable",
        }

    return {
        "translation": translated,
        "definition": (
            f"'{text}' is a French word meaning '{translated}'."
        ),
    }


# === CLI Helpers ===

def _create_detailed_log() -> Path:
    """
    Create the detailed quiz CSV log if it does not already exist.
    """
    LOGS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    detailed_log_path = LOGS_DIR / "fluentique_detailed_log.csv"

    if detailed_log_path.exists():
        return detailed_log_path

    try:
        with detailed_log_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    "question_number",
                    "level",
                    "correct_word",
                    "selected_word",
                    "is_correct",
                    "time_taken",
                ]
            )

    except OSError as exc:
        raise RuntimeError(
            f"Unable to create quiz log at {detailed_log_path}."
        ) from exc

    return detailed_log_path


def _append_detailed_result(
    log_path: Path,
    question_number: int,
    level: str,
    correct_word: str,
    selected_word: str,
    is_correct: bool,
    time_taken: float,
) -> None:
    """
    Append one listening-quiz result to the detailed CSV log.
    """
    try:
        with log_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    question_number,
                    level,
                    correct_word,
                    selected_word,
                    int(is_correct),
                    time_taken,
                ]
            )

    except OSError:
        logger.exception(
            "Failed to write quiz result to %s.",
            log_path,
        )


def _build_challenge_round(
    incorrect_words: list[str],
    asked_questions: set[str],
) -> list[dict]:
    """
    Build a personalised hard-question challenge round.

    Hard questions missed in the main quiz are prioritised. Remaining slots
    are filled with unused hard questions, then other hard questions if needed.
    """
    hard_questions = list(by_level["hard"])

    missed_hard = [
        question
        for question in hard_questions
        if question["correct"] in incorrect_words
    ]

    random.shuffle(missed_hard)

    challenge = missed_hard[:CHALLENGE_LENGTH]
    selected_words = {
        question["correct"]
        for question in challenge
    }

    unused_hard = [
        question
        for question in hard_questions
        if question["correct"] not in asked_questions
        and question["correct"] not in selected_words
    ]

    random.shuffle(unused_hard)

    for question in unused_hard:
        if len(challenge) >= CHALLENGE_LENGTH:
            break

        challenge.append(question)
        selected_words.add(question["correct"])

    if len(challenge) < CHALLENGE_LENGTH:
        remaining_hard = [
            question
            for question in hard_questions
            if question["correct"] not in selected_words
        ]

        random.shuffle(remaining_hard)

        for question in remaining_hard:
            if len(challenge) >= CHALLENGE_LENGTH:
                break

            challenge.append(question)
            selected_words.add(question["correct"])

    return challenge


def _save_score_log(
    score: int,
    main_questions_answered: int,
    challenge_score: int,
    challenge_questions_answered: int,
    incorrect_words: list[str],
) -> None:
    """
    Save a concise summary of the completed CLI quiz.
    """
    LOGS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    score_log_path = LOGS_DIR / "fluentique_score_log.txt"

    try:
        with score_log_path.open(
            "a",
            encoding="utf-8",
        ) as log_file:
            log_file.write(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                f"- Main score: {score}/{main_questions_answered}; "
                f"Challenge: {challenge_score}/"
                f"{challenge_questions_answered}\n"
            )

            if incorrect_words:
                log_file.write(
                    "Incorrect words: "
                    + ", ".join(incorrect_words)
                    + "\n"
                )

            log_file.write("\n")

    except OSError:
        logger.exception(
            "Failed to save quiz score log."
        )


# === Standalone CLI Quiz ===

def run_cli_quiz() -> None:
    """
    Run the original standalone French listening quiz.

    Difficulty moves up after five consecutive correct answers and down after
    three consecutive incorrect answers. The challenge round uses hard
    questions and prioritises hard words missed during the main quiz.
    """
    detailed_log_path = _create_detailed_log()

    score = 0
    incorrect_words: list[str] = []
    asked_questions: set[str] = set()

    current_level = "easy"
    question_count = 0
    correct_streak = 0
    incorrect_streak = 0

    print("Welcome to Fluentique - French Listening Quiz!")

    # === Main Quiz: Adaptive Level Switching ===

    while question_count < MAIN_QUIZ_LENGTH:
        available_questions = [
            question
            for question in by_level[current_level]
            if question["correct"] not in asked_questions
        ]

        if not available_questions:
            print(
                "No more unique questions available in this level."
            )
            break

        question = random.choice(
            available_questions
        )

        asked_questions.add(
            question["correct"]
        )

        correct_word = question["correct"]
        options = question["options"]
        correct_index = question["answer_index"]

        print(
            f"\nQuestion {question_count + 1} "
            f"({current_level.upper()}):"
        )

        play_word_with_polly(
            correct_word
        )

        for index, option in enumerate(
            options,
            start=1,
        ):
            print(
                f"{index}. {option}"
            )

        start_time = time.time()

        while True:
            user_input = input(
                "What word did you hear? "
                "(1-4 or 'r' to replay): "
            ).strip().lower()

            if user_input == "r":
                play_word_with_polly(
                    correct_word
                )
                continue

            try:
                choice = int(user_input) - 1

            except ValueError:
                print(
                    "Invalid input - type a valid option number "
                    "or 'r' to replay."
                )
                continue

            if not 0 <= choice < len(options):
                print(
                    f"Please enter a number between "
                    f"1 and {len(options)}."
                )
                continue

            time_taken = round(
                time.time() - start_time,
                2,
            )

            selected_word = options[choice]
            is_correct = choice == correct_index

            if is_correct:
                print("Correct!")

                score += 1
                correct_streak += 1
                incorrect_streak = 0

            else:
                print("Incorrect!")

                correct_streak = 0
                incorrect_streak += 1
                incorrect_words.append(
                    correct_word
                )

                correct_meaning = translate_word(
                    correct_word
                )

                selected_meaning = translate_word(
                    selected_word
                )

                print(
                    f"'{correct_word}' means: "
                    f"{correct_meaning}"
                )

                print(
                    f"You chose '{selected_word}', "
                    f"which means: {selected_meaning}"
                )

            info = get_word_info(
                correct_word
            )

            print(
                "\nWord Info:\n"
                f"- Definition: {info['definition']}"
            )

            _append_detailed_result(
                detailed_log_path,
                question_count + 1,
                current_level,
                correct_word,
                selected_word,
                is_correct,
                time_taken,
            )

            break

        question_count += 1

        # === Adaptive difficulty logic ===

        if correct_streak >= 5:
            if current_level == "easy":
                current_level = "moderate"

                print(
                    "\nYou are now on MODERATE level!"
                )

            elif current_level == "moderate":
                current_level = "hard"

                print(
                    "\nYou reached the HARD level!"
                )

            correct_streak = 0

        if incorrect_streak >= 3:
            if current_level == "hard":
                current_level = "moderate"

                print(
                    "\nDropping to MODERATE level."
                )

            elif current_level == "moderate":
                current_level = "easy"

                print(
                    "\nDropping to EASY level."
                )

            incorrect_streak = 0

    # === Personalised Challenge Round ===

    print(
        "\nCreating your personalised challenge round..."
    )

    challenge_questions = _build_challenge_round(
        incorrect_words,
        asked_questions,
    )

    challenge_score = 0

    for index, question in enumerate(
        challenge_questions,
        start=1,
    ):
        print(
            f"\nChallenge Question {index}:"
        )

        correct_word = question["correct"]
        options = question["options"]
        answer_index = question["answer_index"]

        play_word_with_polly(
            correct_word
        )

        for option_index, option in enumerate(
            options,
            start=1,
        ):
            print(
                f"{option_index}. {option}"
            )

        while True:
            user_input = input(
                "What word did you hear? "
                "(1-4 or 'r' to replay): "
            ).strip().lower()

            if user_input == "r":
                play_word_with_polly(
                    correct_word
                )
                continue

            try:
                user_choice = int(
                    user_input
                ) - 1

            except ValueError:
                print(
                    "Invalid input - try again."
                )
                continue

            if not 0 <= user_choice < len(options):
                print(
                    "Invalid input - try again."
                )
                continue

            if user_choice == answer_index:
                print("Correct!")
                challenge_score += 1

            else:
                print(
                    "Incorrect. The correct answer was: "
                    f"{correct_word}"
                )

            info = get_word_info(
                correct_word
            )

            print(
                "\nWord Info:\n"
                f"- Definition: {info['definition']}"
            )

            break

    # === Final Summary and Log ===

    challenge_count = len(challenge_questions)
    total_score = score + challenge_score
    total_questions = question_count + challenge_count

    print("\nQuiz Complete!")
    print(
        f"Main Quiz Score: {score}/{question_count}"
    )
    print(
        f"Challenge Round: "
        f"{challenge_score}/{challenge_count}"
    )
    print(
        f"Total Score: {total_score}/{total_questions}"
    )

    if incorrect_words:
        print(
            "\nHere are the words you missed:"
        )

        for word in incorrect_words:
            print(
                f" - {word}"
            )

    else:
        print(
            "You got everything correct! Well done!"
        )

    _save_score_log(
        score,
        question_count,
        challenge_score,
        challenge_count,
        incorrect_words,
    )


if __name__ == "__main__":
    run_cli_quiz()
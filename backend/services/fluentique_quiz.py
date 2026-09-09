# === Imports ===

import csv
import json
import logging
import pickle
import random
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from deep_translator import GoogleTranslator

from backend.utils.tts_utils import play_word_with_polly


# fluentique_quiz.py
# ----------------------
# This file handles the logic for running quizzes within the app.
# It includes functions for dynamically adjusting the quiz difficulty based on
# the user’s performance and tracks quiz results. It also integrates with the backend
# to fetch quiz data and display questions and answers to the user.
#
# Functions:
# - generate_quiz: Generates quiz questions based on the user's current skill level.
# - validate_answer: Checks the user’s answer and adjusts the difficulty if needed.
# - store_results: Saves quiz results and scores for progress tracking.


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent


# === Setup for log saving ===

LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# === Load pretrained ML model and encoders for difficulty prediction ===

MODEL_PATH = BASE_DIR / "models" / "fluentique_predictor.pkl"

try:
    with MODEL_PATH.open("rb") as file:
        model_data = pickle.load(file)
except (OSError, pickle.UnpicklingError) as exc:
    raise RuntimeError(
        f"Unable to load quiz prediction model from {MODEL_PATH}."
    ) from exc

try:
    predictor = model_data["model"]
    le_level = model_data["le_level"]
    le_correct = model_data["le_correct"]
    le_selected = model_data["le_selected"]
except KeyError as exc:
    raise RuntimeError(
        "Quiz prediction model file is missing required components."
    ) from exc


# === Load static French quiz dataset ===

DATA_PATH = BASE_DIR / "data" / "french_quiz_data_aws.json"

try:
    with DATA_PATH.open("r", encoding="utf-8") as file:
        quiz_data = json.load(file)
except (OSError, json.JSONDecodeError) as exc:
    raise RuntimeError(
        f"Unable to load quiz data from {DATA_PATH}."
    ) from exc

if not isinstance(quiz_data, list) or not quiz_data:
    raise RuntimeError(
        "Quiz data must contain a non-empty list of questions."
    )


# === Organise questions by difficulty level ===

by_level = {
    "easy": [
        question
        for question in quiz_data
        if question.get("level") == "easy"
    ],
    "moderate": [
        question
        for question in quiz_data
        if question.get("level") == "moderate"
    ],
    "hard": [
        question
        for question in quiz_data
        if question.get("level") == "hard"
    ],
}


# === Translation + definition utility for feedback ===

def get_word_info(word: str) -> dict[str, str]:
    """
    Translate a French word and generate a simple definition.
    """
    try:
        translated = GoogleTranslator(
            source="fr",
            target="en",
        ).translate(word)

        definition = (
            f"'{word}' is a French word meaning '{translated}'."
        )

        return {
            "translation": translated,
            "definition": definition,
        }

    except Exception:
        logger.exception(
            "Failed to retrieve word information for %s.",
            word,
        )

        return {
            "translation": "Unavailable",
            "definition": "Definition unavailable",
        }


# === Fallback translator (was unused in updated logic) ===

def translate_word(word: str) -> str:
    """
    Translate a French word to English for quiz feedback.
    """
    try:
        return GoogleTranslator(
            source="fr",
            target="en",
        ).translate(word)

    except Exception:
        logger.exception(
            "Failed to translate French word: %s",
            word,
        )

        return "Translation unavailable"


# === Predict difficulty score using ML model based on learner features ===

def predict_difficulty(
    level: str,
    correct_word: str,
    selected_word: str = "",
) -> float:
    """
    Predict quiz difficulty using the pretrained learner model.
    """
    try:
        level_encoded = le_level.transform(
            [level]
        )[0]

        correct_encoded = le_correct.transform(
            [correct_word]
        )[0]

        selected_encoded = (
            le_selected.transform([selected_word])[0]
            if selected_word
            else 0
        )

        # Fixed value retained from the original trained feature pipeline.
        time_taken = 3.0

        features = pd.DataFrame(
            [
                {
                    "level_encoded": level_encoded,
                    "correct_encoded": correct_encoded,
                    "selected_encoded": selected_encoded,
                    "time_taken": time_taken,
                }
            ]
        )

        prediction = predictor.predict_proba(
            features
        )[0][0]

        return float(prediction)

    except Exception:
        logger.exception(
            "Failed to predict difficulty for word %s.",
            correct_word,
        )

        return 0.0


# === Log setup for CSV-based score tracking ===

detailed_log_path = (
    LOGS_DIR / "fluentique_detailed_log.csv"
)

if not detailed_log_path.exists():
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


# === Global quiz state variables ===

score = 0
incorrect_words = []
asked_questions = set()

stage_progress = {
    "easy": 0,
    "moderate": 0,
    "hard": 0,
}

current_level = "easy"
question_count = 0
correct_streak = 0
incorrect_streak = 0


# === Start of CLI-based quiz interaction ===

if __name__ == "__main__":
    print(" Welcome to Fluentique - French Listening Quiz!")

    # === Main Quiz: 20 Questions with Dynamic Level Switching ===

    while question_count < 20:
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
        )  # TTS output

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
                    "Invalid input - type 1, 2, 3, 4 "
                    "or 'r' to replay."
                )
                continue

            if not 0 <= choice < len(options):
                print(
                    f"Please enter a number between "
                    f"1 and {len(options)}."
                )
                continue

            end_time = time.time()

            time_taken = round(
                end_time - start_time,
                2,
            )

            selected_word = options[choice]

            is_correct = (
                1
                if choice == correct_index
                else 0
            )

            if is_correct:
                print("Correct!")

                info = get_word_info(
                    correct_word
                )

                print(
                    "\nWord Info:\n"
                    f"- Definition: {info['definition']}"
                )

                score += 1
                correct_streak += 1
                incorrect_streak = 0
                stage_progress[current_level] += 1

            else:
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

                print("Incorrect!")

                print(
                    f"'{correct_word}' means: "
                    f"{correct_meaning}"
                )

                print(
                    f"You chose '{selected_word}' "
                    f"which means: {selected_meaning}"
                )

                info = get_word_info(
                    correct_word
                )

                print(
                    "\nWord Info:\n"
                    f"- Definition: {info['definition']}"
                )

            # Save result to CSV log
            try:
                with detailed_log_path.open(
                    "a",
                    newline="",
                    encoding="utf-8",
                ) as file:
                    writer = csv.writer(file)

                    writer.writerow(
                        [
                            question_count + 1,
                            current_level,
                            correct_word,
                            selected_word,
                            is_correct,
                            time_taken,
                        ]
                    )

            except OSError:
                logger.exception(
                    "Failed to write quiz result to %s.",
                    detailed_log_path,
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


    # === Challenge Round with ML-based hard word prediction ===

    print(
        "\nCreating your personalised challenge round..."
    )

    challenge_pool = []

    for question in quiz_data:
        if question["level"] == "hard":
            score_estimate = predict_difficulty(
                question["level"],
                question["correct"],
            )

            challenge_pool.append(
                (
                    question,
                    score_estimate,
                )
            )

    challenge_pool = sorted(
        challenge_pool,
        key=lambda item: item[1],
        reverse=True,
    )[:5]

    for index, (question, _) in enumerate(
        challenge_pool,
        start=1,
    ):
        print(
            f"\nChallenge Question {index}:"
        )

        correct_word = question["correct"]
        options = question["options"]
        answer_index = question["answer_index"]

        while True:
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

            user_input = input(
                "What word did you hear? "
                "(1-4 or 'r' to replay): "
            ).strip().lower()

            if user_input == "r":
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
                print(" Correct!")

                info = get_word_info(
                    correct_word
                )

                print(
                    "\nWord Info:\n"
                    f"- Definition: {info['definition']}"
                )

            else:
                print(
                    " Incorrect. The correct answer was: "
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

    print("\n Quiz Complete!")
    print(
        f"Main Quiz Score: {score}/20"
    )
    print(
        "Challenge Round Bonus: 5/5"
    )
    print(
        "Total Score: 25/25"
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
            " You got everything correct! Well done!"
        )

    score_log_path = (
        LOGS_DIR / "fluentique_score_log.txt"
    )

    try:
        with score_log_path.open(
            "a",
            encoding="utf-8",
        ) as log_file:
            log_file.write(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                f"- Score: {score}/20\n"
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
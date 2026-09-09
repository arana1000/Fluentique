import logging
import os
from pathlib import Path

from openai import OpenAI
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from backend.config import OPENAI_API_KEY


# grammar_feedback.py
# ----------------------
# This file handles the generation of grammar feedback for users.
# It uses a pre-trained grammar correction model (e.g. MarianMT) to check
# for errors in user-provided sentences, then uses GPT-3 to explain why the
# sentence was incorrect and how to correct it. This helps users learn from
# their mistakes by providing context-aware feedback.
#
# Functions:
# - correct_grammar: Uses a grammar correction model to fix the user’s sentence.
# - generate_explanation: Uses GPT-3 to explain why a sentence correction was necessary.


logger = logging.getLogger(__name__)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")


# === Load locally fine-tuned grammar correction model (e.g. Helsinki-NLP based) ===

MODEL_DIR = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "grammar_correction_model"
)

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR)
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    str(MODEL_DIR)
)


# === Initialise OpenAI client for explanation generation ===

client = OpenAI(
    api_key=OPENAI_API_KEY
)


def correct_french_sentence(
    input_text: str,
    max_length: int = 128,
) -> str:
    """
    Runs the user's French sentence through the fine-tuned correction model.

    Args:
        input_text (str): The raw user sentence in French.
        max_length (int): The maximum number of tokens allowed in the output.

    Returns:
        str: The corrected version of the input sentence.
    """
    inputs = tokenizer.encode(
        input_text,
        return_tensors="pt",
        truncation=True,
    )

    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_beams=5,
        early_stopping=True,
    )

    corrected_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    return corrected_text.strip()


def generate_feedback(
    original: str,
    corrected: str,
    username: str = "the learner",
) -> str:
    """
    Uses GPT to generate a natural explanation for the grammar corrections.

    Args:
        original (str): The original user input sentence.
        corrected (str): The corrected version returned by the model.
        username (str): The user's display name, used to personalise the explanation.

    Returns:
        str: A friendly, concise explanation of the correction.
    """
    prompt = (
        f'{username} wrote this French sentence: "{original}".\n'
        f'The correct version is: "{corrected}".\n\n'
        "Please explain what mistakes were made and how they were corrected.\n"
        "Be kind, clear, and concise as if you're helping someone learning French.\n"
        "Include grammar points if possible "
        "(e.g. verb form, gender agreement, etc.)."
    )

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.4,
            max_tokens=200,
        )

        content = response.choices[0].message.content

        if not content:
            return "Explanation unavailable."

        return content.strip()

    except Exception:
        logger.exception(
            "Failed to generate grammar feedback with OpenAI."
        )

        return "Explanation unavailable."


def correct_and_explain(
    user_input: str,
    username: str = "the learner",
) -> dict:
    """
    Pipeline that both corrects a user's sentence and explains the changes.

    Args:
        user_input (str): The raw sentence submitted by the learner.
        username (str): The learner's display name (used in the explanation).

    Returns:
        dict: A dictionary containing the original input, the corrected sentence,
        and a grammar explanation.
    """
    corrected = correct_french_sentence(
        user_input
    )

    explanation = generate_feedback(
        user_input,
        corrected,
        username=username,
    )

    return {
        "input": user_input,
        "corrected": corrected,
        "explanation": explanation,
    }


if __name__ == "__main__":
    """
    Test run for direct CLI usage. Demonstrates correction and explanation output
    for a sample sentence.
    """
    test_input = "Tu est Jane"

    result = correct_and_explain(
        test_input,
        username="Lina",
    )

    print("Original:", result["input"])
    print("Corrected:", result["corrected"])
    print("Explanation:", result["explanation"])
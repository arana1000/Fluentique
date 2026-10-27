import logging

from openai import OpenAI, OpenAIError

from backend.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
)
from backend.utils.openai_utils import describe_openai_error


# grammar_feedback.py
# ----------------------
# This file handles AI-powered French grammar correction and explanation.
# The portfolio/runtime version uses OpenAI rather than loading a locally
# fine-tuned grammar model, which keeps the production application lighter
# and simpler to deploy while preserving the original service interface.
#
# Functions:
# - correct_french_sentence: Corrects a learner's French sentence using OpenAI.
# - generate_feedback: Explains the difference between the original and corrected sentence.
# - correct_and_explain: Runs correction and explanation as a single service pipeline.


logger = logging.getLogger(__name__)

client = OpenAI(
    api_key=OPENAI_API_KEY
)


def _get_response_text(response) -> str:
    """
    Extract and validate text from an OpenAI chat completion response.
    """
    content = response.choices[0].message.content

    if not content or not content.strip():
        raise RuntimeError(
            "OpenAI returned an empty response."
        )

    return content.strip()


def correct_french_sentence(
    input_text: str,
    max_length: int = 128,
) -> str:
    """
    Correct a French sentence using OpenAI while preserving its intended
    meaning and wording as closely as possible.

    Args:
        input_text (str): The raw user sentence in French.
        max_length (int): Retained for backwards compatibility with the
            previous local-model function signature.

    Returns:
        str: The corrected version of the input sentence.
    """
    del max_length  # Retained only for backwards-compatible callers.

    text = input_text.strip()

    if not text:
        raise ValueError(
            "Input text cannot be empty."
        )

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a French grammar correction assistant. "
                        "Correct grammar, spelling, accents, conjugation, "
                        "agreement, and word order only when necessary. "
                        "Preserve the learner's intended meaning and wording "
                        "as closely as possible. "
                        "If the sentence is already correct, return it unchanged. "
                        "Return only the corrected French sentence with no "
                        "explanation, labels, quotation marks, or markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            temperature=0.0,
        )

        return _get_response_text(
            response
        )

    except OpenAIError as exc:
        detail = describe_openai_error(exc)
        logger.warning(
            "Grammar correction unavailable: %s",
            detail,
        )
        raise RuntimeError(detail) from exc

    except RuntimeError as exc:
        logger.exception(
            "OpenAI returned an invalid grammar-correction response."
        )
        raise RuntimeError(
            "AI grammar correction returned an invalid response."
        ) from exc


def generate_feedback(
    original: str,
    corrected: str,
    username: str = "the learner",
) -> str:
    """
    Generate a natural explanation for a French grammar correction.

    Args:
        original (str): The original user input sentence.
        corrected (str): The corrected French sentence.
        username (str): Learner display name used to personalise feedback.

    Returns:
        str: A friendly, concise explanation of the correction.
    """
    original = original.strip()
    corrected = corrected.strip()

    if not original or not corrected:
        logger.warning(
            "Grammar feedback requested with missing original or corrected text."
        )
        return "Explanation unavailable."

    if original == corrected:
        return (
            "No grammar correction was needed. "
            "The sentence is already correct."
        )

    prompt = (
        f'{username} wrote this French sentence: "{original}".\n'
        f'The correct version is: "{corrected}".\n\n'
        "Explain what was wrong and why the corrected version is better. "
        "Be kind, clear, and concise for an English-speaking French learner. "
        "Mention useful grammar points such as conjugation, gender, agreement, "
        "articles, accents, spelling, or word order only when relevant. "
        "Do not invent mistakes that are not present."
    )

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a patient French language tutor. "
                        "Explain French grammar corrections accurately and "
                        "concisely in English."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.3,
        )

        return _get_response_text(
            response
        )

    except OpenAIError as exc:
        detail = describe_openai_error(exc)
        logger.warning(
            "Grammar explanation unavailable: %s",
            detail,
        )
        return "AI explanation unavailable. Check OpenAI API access."

    except RuntimeError:
        logger.exception(
            "OpenAI returned an invalid grammar-feedback response."
        )
        return "AI explanation unavailable."


def correct_and_explain(
    user_input: str,
    username: str = "the learner",
) -> dict:
    """
    Correct a learner's French sentence and explain any changes.

    Args:
        user_input (str): The raw sentence submitted by the learner.
        username (str): Learner display name used in the explanation.

    Returns:
        dict: The original input, corrected sentence, and grammar explanation.
    """
    original = user_input.strip()

    if not original:
        raise ValueError(
            "User input cannot be empty."
        )

    corrected = correct_french_sentence(
        original
    )

    explanation = generate_feedback(
        original,
        corrected,
        username=username,
    )

    return {
        "input": original,
        "corrected": corrected,
        "explanation": explanation,
    }


if __name__ == "__main__":
    """
    Simple direct-run example for manually testing correction and explanation.
    """
    test_input = "Tu est Jane"

    result = correct_and_explain(
        test_input,
        username="Lina",
    )

    print(
        "Original:",
        result["input"],
    )
    print(
        "Corrected:",
        result["corrected"],
    )
    print(
        "Explanation:",
        result["explanation"],
    )

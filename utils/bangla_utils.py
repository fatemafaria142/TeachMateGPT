"""
Bangla-specific utility helpers.
"""
from __future__ import annotations

import re
import unicodedata


BANGLA_DIGITS = "০১২৩৪৫৬৭৮৯"
ENGLISH_DIGITS = "0123456789"

TRANS_TABLE = str.maketrans(ENGLISH_DIGITS, BANGLA_DIGITS)
TRANS_TABLE_REVERSE = str.maketrans(BANGLA_DIGITS, ENGLISH_DIGITS)


def to_bangla_digits(text: str) -> str:
    """Convert English digits to Bangla digits."""
    return text.translate(TRANS_TABLE)


def to_english_digits(text: str) -> str:
    """Convert Bangla digits to English digits."""
    return text.translate(TRANS_TABLE_REVERSE)


def is_bangla(text: str) -> bool:
    """Return True if the text contains Bangla Unicode characters."""
    for ch in text:
        if "\u0980" <= ch <= "\u09FF":
            return True
    return False


def format_mcq(
    question: str,
    options: list[str],
    correct_index: int,
    show_answer: bool = False,
) -> str:
    """Format an MCQ in Bangla display format."""
    labels = ["ক", "খ", "গ", "ঘ"]
    lines = [f"**প্রশ্ন:** {question}", ""]
    for i, opt in enumerate(options):
        lines.append(f"({labels[i]}) {opt}")
    if show_answer:
        lines.append(f"\n**সঠিক উত্তর:** ({labels[correct_index]}) {options[correct_index]}")
    return "\n".join(lines)


def level_label_bangla(level: str) -> str:
    """Return Bangla label for a given level string."""
    mapping = {
        "beginner":     "শিক্ষানবিস",
        "intermediate": "মধ্যবর্তী",
        "advanced":     "উন্নত",
    }
    return mapping.get(level, level)


def clean_bangla_text(text: str) -> str:
    """Normalize Bangla Unicode text (NFC form)."""
    return unicodedata.normalize("NFC", text.strip())

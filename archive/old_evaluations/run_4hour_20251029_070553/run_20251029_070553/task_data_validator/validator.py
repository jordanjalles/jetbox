"""Utility validation functions.

This module provides simple validation helpers for email, phone number and age.
"""

import re
from datetime import datetime

# Regular expression for a basic email validation
EMAIL_REGEX = re.compile(
    r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
)

# Regular expression for a basic phone number validation (digits, optional +, spaces, dashes, parentheses)
PHONE_REGEX = re.compile(
    r"^\+?\d{1,3}?[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}$"
)


def validate_email(email: str) -> bool:
    """Return True if *email* looks like a valid email address.

    The check is intentionally simple and suitable for most use‑cases.
    """
    if not isinstance(email, str):
        return False
    return bool(EMAIL_REGEX.match(email.strip()))


def validate_phone(phone: str) -> bool:
    """Return True if *phone* looks like a valid phone number.

    Accepts international format with optional leading ``+`` and common separators.
    """
    if not isinstance(phone, str):
        return False
    return bool(PHONE_REGEX.match(phone.strip()))


def validate_age(age: int | str) -> bool:
    """Return True if *age* is a positive integer and less than 150.

    Accepts either an ``int`` or a string that can be converted to an int.
    """
    try:
        age_int = int(age)
    except (TypeError, ValueError):
        return False
    return 0 < age_int < 150

# If run as a script, perform a quick demo
if __name__ == "__main__":
    examples = [
        ("validate_email", "test@example.com", True),
        ("validate_email", "invalid-email", False),
        ("validate_phone", "+1 (555) 123-4567", True),
        ("validate_phone", "12345", False),
        ("validate_age", 30, True),
        ("validate_age", "200", False),
    ]
    for func, value, expected in examples:
        result = globals()[func](value)
        print(f"{func}({value!r}) -> {result} (expected {expected})")

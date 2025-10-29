"""Utility module for validating email, phone, and URL strings.

The functions use regular expressions to perform basic validation. They are
intended for quick checks and not for exhaustive validation. For production
use, consider using dedicated libraries such as `email_validator` or
`phonenumbers`.
"""

import re

# Regular expression patterns
# Email pattern: simple RFC 5322 compliant-ish pattern
EMAIL_REGEX = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"  # local part
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"  # domain label start
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*"  # subdomains
    r"\.[A-Za-z]{2,}$"  # TLD
)

# Phone pattern: accepts digits, spaces, dashes, parentheses, and plus sign
PHONE_REGEX = re.compile(r"^\+?\d[\d\s\-()]{7,}\d$")

# URL pattern: basic http/https/ftp scheme with optional www and path
URL_REGEX = re.compile(
    r"^(https?|ftp)://"  # scheme
    r"(?:(?:[A-Za-z0-9\-._~%]+)@)?"  # optional userinfo
    r"(?:(?:[A-Za-z0-9\-._~%]+\.)+[A-Za-z]{2,}|"  # domain
    r"\d{1,3}(?:\.\d{1,3}){3})"  # or IPv4
    r"(?::\d{1,5})?"  # optional port
    r"(?:/[^\s]*)?$",  # optional path
    re.IGNORECASE,
)


def validate_email(email: str) -> bool:
    """Return True if *email* matches a basic email pattern.

    Parameters
    ----------
    email: str
        The email address to validate.

    Returns
    -------
    bool
        ``True`` if the email matches the pattern, otherwise ``False``.
    """
    if not isinstance(email, str):
        return False
    return bool(EMAIL_REGEX.match(email))


def validate_phone(phone: str) -> bool:
    """Return True if *phone* matches a basic international phone pattern.

    The pattern allows an optional leading ``+`` followed by digits and
    common separators such as spaces, dashes, or parentheses. It requires at
    least 8 digits in total.
    """
    if not isinstance(phone, str):
        return False
    return bool(PHONE_REGEX.match(phone))


def validate_url(url: str) -> bool:
    """Return True if *url* matches a basic URL pattern.

    The pattern supports http, https, and ftp schemes, optional userinfo,
    domain names or IPv4 addresses, optional port, and an optional path.
    """
    if not isinstance(url, str):
        return False
    return bool(URL_REGEX.match(url))

# If run as a script, perform a simple demo
if __name__ == "__main__":
    examples = {
        "email": ["test@example.com", "invalid-email", "user.name+tag+sorting@example.co.uk"],
        "phone": ["+1-800-555-0199", "(800) 555 0199", "12345", "+44 20 7946 0958"],
        "url": ["https://www.example.com", "ftp://ftp.example.org/resource.txt", "http://256.256.256.256", "not a url"],
    }
    for kind, vals in examples.items():
        print(f"\n{kind.capitalize()} validation:")
        for v in vals:
            func = globals()[f"validate_{kind}"]
            print(f"  {v:30} -> {func(v)}")
"""

def capitalize(text: str) -> str:
    """Return the string with the first character capitalized and the rest lowercased."""
    if not text:
        return text
    return text[0].upper() + text[1:].lower()
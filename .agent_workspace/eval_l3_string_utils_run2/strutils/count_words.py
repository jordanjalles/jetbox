def count_words(text: str) -> int:
    """Return the number of words in the string.
    Words are separated by whitespace.
    """
    if not text:
        return 0
    return len(text.split())
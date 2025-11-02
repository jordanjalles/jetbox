"""Count words in a string.

This function splits the string on whitespace and returns the number of
words. Empty strings or strings containing only whitespace return 0.

Examples
--------
>>> count_words("hello world")
2
>>> count_words("  leading and trailing  ")
3
>>> count_words("")
0
"""

def count_words(s: str) -> int:
    return len([w for w in s.split() if w])

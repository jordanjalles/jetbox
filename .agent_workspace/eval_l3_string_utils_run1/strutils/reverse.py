"""Reverse a string.

This function returns a new string that is the reverse of the input.

Examples
--------
>>> reverse("hello")
'olleh'
>>> reverse("")
''
"""

def reverse(s: str) -> str:
    return s[::-1]

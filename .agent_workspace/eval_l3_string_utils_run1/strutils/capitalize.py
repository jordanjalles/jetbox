"""Capitalize the first letter of a string.

This function takes a string and returns a new string with the first
character capitalized and the rest unchanged.

Examples
--------
>>> capitalize("hello")
'Hello'
>>> capitalize("Hello")
'Hello'
>>> capitalize("")
''
"""

def capitalize(s: str) -> str:
    if not s:
        return s
    return s[0].upper() + s[1:]

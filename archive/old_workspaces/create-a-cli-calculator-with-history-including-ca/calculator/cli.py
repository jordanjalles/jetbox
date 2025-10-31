"""CLI implementation for the calculator package.

The :func:`main` function provides a simple REPL that supports the
following commands:

* ``quit`` – exit the program
* ``history`` – list all calculations performed in the current session
* ``clear`` – clear the history

Any other input is treated as an arithmetic expression and evaluated.
"""

from __future__ import annotations

from . import Calculator


def _print_welcome() -> None:
    print("Simple CLI Calculator")
    print("Type 'quit' to exit, 'history' to view history, 'clear' to clear history.")


def main() -> None:
    calc = Calculator()
    _print_welcome()
    while True:
        try:
            expr = input(">>> ").strip()
        except EOFError:
            print()
            break
        if not expr:
            continue
        if expr.lower() == "quit":
            print("Goodbye!")
            break
        if expr.lower() == "history":
            for idx, (exp, res) in enumerate(calc.history, 1):
                print(f"{idx}: {exp} = {res}")
            continue
        if expr.lower() == "clear":
            calc.clear_history()
            print("History cleared.")
            continue
        try:
            result = calc.calculate(expr)
            print(result)
        except ValueError as err:
            print(err)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

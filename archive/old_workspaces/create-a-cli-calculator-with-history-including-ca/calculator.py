"""CLI Calculator with history.

This module defines a Calculator class that can evaluate simple arithmetic expressions
and keep a history of all calculations performed. It also provides a simple
interactive command line interface.
"""

from __future__ import annotations

class Calculator:
    """Simple calculator that evaluates arithmetic expressions.

    The calculator uses Python's ``eval`` with a restricted globals dictionary
    to avoid executing arbitrary code. Only basic arithmetic operators are
    supported.
    """

    def __init__(self) -> None:
        self._history: list[tuple[str, float]] = []

    def calculate(self, expression: str) -> float:
        """Evaluate *expression* and store the result in history.

        Parameters
        ----------
        expression:
            A string containing a Python expression that uses only arithmetic
            operators.

        Returns
        -------
        float
            The numeric result of the expression.

        Raises
        ------
        ValueError
            If the expression is invalid or cannot be evaluated.
        """
        try:
            # ``eval`` with no builtins and an empty locals dict.
            result = eval(expression, {"__builtins__": None}, {})
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid expression: {expression}") from exc
        # Store in history
        self._history.append((expression, result))
        return result

    @property
    def history(self) -> list[tuple[str, float]]:
        """Return a copy of the calculation history."""
        return list(self._history)

    def clear_history(self) -> None:
        """Clear the calculation history."""
        self._history.clear()


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------

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

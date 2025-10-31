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

# Interactive CLI is omitted from the package to keep tests focused on logic.

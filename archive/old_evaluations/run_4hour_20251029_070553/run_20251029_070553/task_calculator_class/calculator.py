"""Calculator module.

This module defines a Calculator class that supports basic arithmetic
operations and keeps a history of all operations performed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Operation:
    """Represents a single arithmetic operation.

    Attributes
    ----------
    op: str
        The operation name: ``add``, ``subtract``, ``multiply`` or ``divide``.
    operands: Tuple[float, float]
        The operands used for the operation.
    result: float
        The result of the operation.
    """

    op: str
    operands: Tuple[float, float]
    result: float

    def __str__(self) -> str:
        a, b = self.operands
        return f"{self.op}({a}, {b}) = {self.result}"


class Calculator:
    """Simple calculator with history tracking.

    The calculator supports addition, subtraction, multiplication and
    division.  All operations are stored in a history list which can be
    inspected or cleared.
    """

    def __init__(self) -> None:
        self._history: List[Operation] = []

    @property
    def history(self) -> List[Operation]:
        """Return a copy of the operation history."""
        return list(self._history)

    def _record(self, op: str, a: float, b: float, result: float) -> None:
        self._history.append(Operation(op, (a, b), result))

    def add(self, a: float, b: float) -> float:
        result = a + b
        self._record("add", a, b, result)
        return result

    def subtract(self, a: float, b: float) -> float:
        result = a - b
        self._record("subtract", a, b, result)
        return result

    def multiply(self, a: float, b: float) -> float:
        result = a * b
        self._record("multiply", a, b, result)
        return result

    def divide(self, a: float, b: float) -> float:
        if b == 0:
            raise ZeroDivisionError("division by zero")
        result = a / b
        self._record("divide", a, b, result)
        return result

    def clear_history(self) -> None:
        """Clear the operation history."""
        self._history.clear()

    def __repr__(self) -> str:
        return f"<Calculator history={len(self._history)} ops>"

# Simple test harness (not using pytest, just for quick manual checks)
if __name__ == "__main__":
    calc = Calculator()
    print(calc.add(2, 3))
    print(calc.subtract(5, 2))
    print(calc.multiply(4, 3))
    print(calc.divide(10, 2))
    print("History:")
    for op in calc.history:
        print(op)

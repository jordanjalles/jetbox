#!/usr/bin/env python3
"""CLI calculator with history support.

Usage:
  calc.py add 2 3
  calc.py sub 5 2
  calc.py mul 4 3
  calc.py div 10 2
  calc.py history

The history command prints all previous calculations.
"""

import argparse
import sys
import os
from pathlib import Path

HISTORY_FILE = Path("calc_history.txt")


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def mul(a, b):
    return a * b


def div(a, b):
    if b == 0:
        raise ZeroDivisionError("division by zero")
    return a / b


OPERATIONS = {
    "add": add,
    "sub": sub,
    "mul": mul,
    "div": div,
}


def record_history(op, args, result):
    line = f"{op} {' '.join(map(str, args))} = {result}\n"
    HISTORY_FILE.write_text(HISTORY_FILE.read_text() + line, encoding="utf-8") if HISTORY_FILE.exists() else HISTORY_FILE.write_text(line, encoding="utf-8")


def show_history():
    if not HISTORY_FILE.exists():
        print("No history.")
        return
    print(HISTORY_FILE.read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Simple CLI calculator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Operation parsers
    for op in OPERATIONS:
        p = subparsers.add_parser(op, help=f"{op} two numbers")
        p.add_argument("a", type=float, help="first operand")
        p.add_argument("b", type=float, help="second operand")

    # History parser
    subparsers.add_parser("history", help="Show calculation history")

    args = parser.parse_args(argv)

    if args.command == "history":
        show_history()
        return

    func = OPERATIONS[args.command]
    try:
        result = func(args.a, args.b)
    except ZeroDivisionError as e:
        print(e)
        sys.exit(1)

    print(result)
    record_history(args.command, [args.a, args.b], result)


if __name__ == "__main__":
    main()

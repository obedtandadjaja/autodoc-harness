"""Tiny CLI entry point - fixture content for autodoc-harness's own opt-in
end-to-end test. Deliberately has clear green/yellow/red branches to document.
"""

import sys

from calculator import divide


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: main.py <a> <b>", file=sys.stderr)
        return 2

    try:
        a, b = float(sys.argv[1]), float(sys.argv[2])
    except ValueError:
        print("error: arguments must be numbers", file=sys.stderr)
        return 1

    try:
        result = divide(a, b)
    except ZeroDivisionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

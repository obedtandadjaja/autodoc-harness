"""A tiny division helper with deliberately distinct success, edge-case, and
error branches - fixture content for autodoc-harness's own opt-in end-to-end test.
"""

import warnings

PRECISION_WARNING_THRESHOLD = 1e-6


def divide(a: float, b: float) -> float:
    """Divide a by b.

    - Green (happy path): b is a normal non-zero divisor - returns a / b.
    - Yellow (edge case): b is extremely small (but non-zero) - the result may
      lose precision, so a warning is emitted before returning it.
    - Red (error path): b is exactly zero - raises ZeroDivisionError with a
      descriptive message rather than letting Python's own error propagate.
    """
    if b == 0:
        raise ZeroDivisionError(f"cannot divide {a} by zero")

    if abs(b) < PRECISION_WARNING_THRESHOLD:
        warnings.warn(
            f"divisor {b} is extremely small; result may lose precision",
            stacklevel=2,
        )

    return a / b

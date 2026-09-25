"""Low-level numerical primitives (normal CDF/PDF, Brent root finding).

All failures are raised as :class:`PricingError` with distinguishable codes;
raw exceptions from numerical work never escape the service boundary.
"""
from __future__ import annotations

import math
from collections.abc import Callable

from app.errors import ErrorCode, PricingError

_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    if not math.isfinite(x):
        raise PricingError(
            ErrorCode.NUMERICAL_INSTABILITY,
            "normal CDF received a non-finite argument",
            {"argument": repr(x)},
        )
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


def norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    if not math.isfinite(x):
        raise PricingError(
            ErrorCode.NUMERICAL_INSTABILITY,
            "normal PDF received a non-finite argument",
            {"argument": repr(x)},
        )
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def brentq(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """Find a root of ``f`` on ``[a, b]`` using Brent's method.

    Raises:
        PricingError(CALIBRATION_NO_SOLUTION): bracket has no sign change.
        PricingError(CALIBRATION_DID_NOT_CONVERGE): iteration budget exhausted.
        PricingError(NUMERICAL_INSTABILITY): non-finite inputs/function values.
    """
    if not (math.isfinite(a) and math.isfinite(b)):
        raise PricingError(
            ErrorCode.NUMERICAL_INSTABILITY,
            "root bracket endpoints must be finite",
            {"a": repr(a), "b": repr(b)},
        )
    fa = f(a)
    fb = f(b)
    if not (math.isfinite(fa) and math.isfinite(fb)):
        raise PricingError(
            ErrorCode.NUMERICAL_INSTABILITY,
            "objective returned a non-finite value at the bracket",
            {"f(a)": repr(fa), "f(b)": repr(fb)},
        )
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise PricingError(
            ErrorCode.CALIBRATION_NO_SOLUTION,
            "no root: objective has the same sign at both bracket endpoints",
            {"a": a, "b": b, "f(a)": fa, "f(b)": fb},
        )

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c, fc = a, fa
    d = 0.0
    mflag = True

    for _ in range(max_iter):
        if fb == 0.0 or abs(b - a) < tol:
            return b

        if fa != fc and fb != fc:
            # Inverse quadratic interpolation.
            s = (
                a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
            )
        else:
            # Secant step.
            s = b - fb * (b - a) / (fb - fa)

        lo = (3.0 * a + b) / 4.0
        between = (lo < s < b) or (b < s < lo)
        # Standard Brent acceptability conditions; otherwise bisect.
        use_bisection = (
            not between
            or (mflag and abs(s - b) >= abs(b - c) / 2.0)
            or (not mflag and abs(s - b) >= abs(c - d) / 2.0)
            or (mflag and abs(b - c) < tol)
            or (not mflag and abs(c - d) < tol)
        )
        if use_bisection:
            s = (a + b) / 2.0
            mflag = True
        else:
            mflag = False

        fs = f(s)
        if not math.isfinite(fs):
            raise PricingError(
                ErrorCode.NUMERICAL_INSTABILITY,
                "objective returned a non-finite value during iteration",
                {"x": repr(s), "f(x)": repr(fs)},
            )

        d, c = c, b
        if fa * fs < 0.0:
            b, fb = s, fs
        else:
            a, fa = s, fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        c, fc = a, fa

    raise PricingError(
        ErrorCode.CALIBRATION_DID_NOT_CONVERGE,
        "root finder exhausted the iteration budget without reaching tolerance",
        {"max_iter": max_iter, "tolerance": tol, "interval": [a, b]},
    )

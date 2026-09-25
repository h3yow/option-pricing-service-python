"""Analytic Black-Scholes-Merton pricing for European options.

Continuous dividend yield ``q`` is supported. Greeks conventions:

* ``delta``  : dPrice/dSpot
* ``gamma``  : d2Price/dSpot2
* ``vega``   : dPrice/d(sigma)        (per 1.0, i.e. 100 vol points)
* ``theta``  : dPrice/dt per year (calendar-time decay; = -dPrice/dT)
* ``rho``    : dPrice/dr              (per 1.0, i.e. 100 rate points)
"""
from __future__ import annotations

import math

from app.errors import ErrorCode, PricingError
from app.numerics import brentq, norm_cdf, norm_pdf

_VOL_LO = 1e-6
_VOL_HI = 5.0
_PRICE_TOL = 1e-10


def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float):
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2


def _ensure_finite(values: dict[str, float]) -> None:
    for name, value in values.items():
        if not math.isfinite(value):
            raise PricingError(
                ErrorCode.NUMERICAL_INSTABILITY,
                f"non-finite Black-Scholes input: {name}",
                {name: repr(value)},
            )


def bs_price(
    option_type: str,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
) -> float:
    _ensure_finite(
        {"S": S, "K": K, "T": T, "r": r, "q": q, "sigma": sigma}
    )
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_k = K * math.exp(-r * T)
    fwd_stock = S * math.exp(-q * T)
    if option_type == "call":
        price = fwd_stock * norm_cdf(d1) - disc_k * norm_cdf(d2)
    elif option_type == "put":
        price = disc_k * norm_cdf(-d2) - fwd_stock * norm_cdf(-d1)
    else:  # pragma: no cover - guarded by validation upstream
        raise PricingError(
            ErrorCode.INVALID_PARAMETER,
            f"unknown option type {option_type!r}",
        )
    if not math.isfinite(price):
        raise PricingError(
            ErrorCode.NUMERICAL_INSTABILITY,
            "Black-Scholes price is not finite",
            {"price": repr(price)},
        )
    return max(price, 0.0)


def bs_greeks(
    option_type: str,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
) -> dict[str, float]:
    _ensure_finite({"S": S, "K": K, "T": T, "r": r, "q": q, "sigma": sigma})
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    sqrt_t = math.sqrt(T)
    disc_k = K * math.exp(-r * T)
    fwd_stock = S * math.exp(-q * T)
    pdf_d1 = norm_pdf(d1)

    gamma = math.exp(-q * T) * pdf_d1 / (S * sigma * sqrt_t)
    vega = fwd_stock * pdf_d1 * sqrt_t

    if option_type == "call":
        delta = math.exp(-q * T) * norm_cdf(d1)
        theta = (
            -fwd_stock * pdf_d1 * sigma / (2.0 * sqrt_t)
            - r * disc_k * norm_cdf(d2)
            + q * fwd_stock * norm_cdf(d1)
        )
        rho = disc_k * T * norm_cdf(d2)
    elif option_type == "put":
        delta = math.exp(-q * T) * (norm_cdf(d1) - 1.0)
        theta = (
            -fwd_stock * pdf_d1 * sigma / (2.0 * sqrt_t)
            + r * disc_k * norm_cdf(-d2)
            - q * fwd_stock * norm_cdf(-d1)
        )
        rho = -disc_k * T * norm_cdf(-d2)
    else:  # pragma: no cover - guarded by validation upstream
        raise PricingError(
            ErrorCode.INVALID_PARAMETER,
            f"unknown option type {option_type!r}",
        )

    greeks = {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }
    for name, value in greeks.items():
        if not math.isfinite(value):
            raise PricingError(
                ErrorCode.NUMERICAL_INSTABILITY,
                f"Black-Scholes greek {name} is not finite",
                {name: repr(value)},
            )
    return greeks


def implied_vol(
    option_type: str,
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
) -> float:
    """Back out the implied volatility of a European premium quote.

    Raises:
        PricingError(CALIBRATION_NO_SOLUTION): premium is outside the
            model-implied no-arbitrage range (intrinsic .. asymptotic max).
        PricingError(CALIBRATION_DID_NOT_CONVERGE): solver failed to converge.
    """
    _ensure_finite({"price": price, "S": S, "K": K, "T": T, "r": r, "q": q})
    if price <= 0.0:
        raise PricingError(
            ErrorCode.CALIBRATION_NO_SOLUTION,
            "quoted premium must be strictly positive for a finite implied vol",
            {"premium": price},
        )

    disc_k = K * math.exp(-r * T)
    fwd_stock = S * math.exp(-q * T)
    if option_type == "call":
        intrinsic = max(fwd_stock - disc_k, 0.0)
        asymptotic_max = fwd_stock
    elif option_type == "put":
        intrinsic = max(disc_k - fwd_stock, 0.0)
        asymptotic_max = disc_k
    else:  # pragma: no cover - guarded by validation upstream
        raise PricingError(
            ErrorCode.INVALID_PARAMETER,
            f"unknown option type {option_type!r}",
        )

    if price <= intrinsic + _PRICE_TOL:
        raise PricingError(
            ErrorCode.CALIBRATION_NO_SOLUTION,
            "quoted premium is at or below intrinsic value; "
            "no strictly positive implied volatility exists",
            {
                "premium": price,
                "intrinsic": intrinsic,
                "option_type": option_type,
            },
        )
    if price >= asymptotic_max - _PRICE_TOL:
        raise PricingError(
            ErrorCode.CALIBRATION_NO_SOLUTION,
            "quoted premium is at or above the asymptotic maximum "
            "as volatility tends to infinity",
            {
                "premium": price,
                "asymptotic_max": asymptotic_max,
                "option_type": option_type,
            },
        )

    def objective(sigma: float) -> float:
        return bs_price(option_type, S, K, T, r, q, sigma) - price

    # Monotonicity of price in vol guarantees a unique root inside the range.
    return brentq(objective, _VOL_LO, _VOL_HI, tol=1e-9, max_iter=100)

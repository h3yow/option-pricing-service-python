"""Reiner-Rubinstein analytic formulas for continuously-monitored
single-barrier European options (Merton, 1973; Reiner & Rubinstein, 1991).

Supported barriers: up-and-in / up-and-out / down-and-in / down-and-out,
each as call or put. A cash ``rebate`` is paid *at the knock-out time* for
knock-out options, and at expiry for knock-in options that never knock in
(the market-standard conventions).

All 16 payoff/branch combinations plus the two rebate variants were verified
against Brownian-bridge Monte Carlo (see tests/test_barrier.py).
"""
from __future__ import annotations

import math

from app.errors import ErrorCode, PricingError
from app.numerics import norm_cdf


def _ensure_finite(values: dict[str, float]) -> None:
    for name, value in values.items():
        if not math.isfinite(value):
            raise PricingError(
                ErrorCode.NUMERICAL_INSTABILITY,
                f"non-finite barrier input: {name}",
                {name: repr(value)},
            )


def barrier_price(
    barrier_type: str,
    option_type: str,
    S: float,
    K: float,
    H: float,
    rebate: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
) -> float:
    _ensure_finite(
        {"S": S, "K": K, "H": H, "rebate": rebate,
         "T": T, "r": r, "q": q, "sigma": sigma}
    )

    sigma2 = sigma * sigma
    sqrt_t = math.sqrt(T)
    sigma_sqrt_t = sigma * sqrt_t
    # mu follows the payoff convention used by terms A..E.
    mu = (r - q - 0.5 * sigma2) / sigma2

    x1 = math.log(S / K) / sigma_sqrt_t + (1.0 + mu) * sigma_sqrt_t
    x2 = math.log(S / H) / sigma_sqrt_t + (1.0 + mu) * sigma_sqrt_t
    y1 = math.log(H * H / (S * K)) / sigma_sqrt_t + (1.0 + mu) * sigma_sqrt_t
    y2 = math.log(H / S) / sigma_sqrt_t + (1.0 + mu) * sigma_sqrt_t

    phi = 1.0 if option_type == "call" else -1.0
    is_call = option_type == "call"
    is_down = barrier_type.startswith("down")
    # C/D payoff terms: down-call uses the +1 sign, all puts use -1.
    eta_cd = 1.0 if (is_call and is_down) else -1.0
    # E/F rebate terms: +1 on the live side for down barriers, -1 for up.
    eta_rb = 1.0 if is_down else -1.0

    S_qt = S * math.exp(-q * T)
    K_rt = K * math.exp(-r * T)
    h2mu = (H / S) ** (2.0 * mu)
    h2mu2 = (H / S) ** (2.0 * mu + 2.0)

    def term_A() -> float:
        return phi * S_qt * norm_cdf(phi * x1) - phi * K_rt * norm_cdf(
            phi * (x1 - sigma_sqrt_t)
        )

    def term_B() -> float:
        return phi * S_qt * norm_cdf(phi * x2) - phi * K_rt * norm_cdf(
            phi * (x2 - sigma_sqrt_t)
        )

    def term_C() -> float:
        return (
            phi * S_qt * h2mu2 * norm_cdf(eta_cd * y1)
            - phi * K_rt * h2mu * norm_cdf(eta_cd * (y1 - sigma_sqrt_t))
        )

    def term_D() -> float:
        return (
            phi * S_qt * h2mu2 * norm_cdf(eta_cd * y2)
            - phi * K_rt * h2mu * norm_cdf(eta_cd * (y2 - sigma_sqrt_t))
        )

    def term_E() -> float:
        # Knock-in rebate, paid at expiry if the barrier is never touched.
        return rebate * math.exp(-r * T) * (
            norm_cdf(eta_rb * (x2 - sigma_sqrt_t))
            - h2mu * norm_cdf(eta_rb * (y2 - sigma_sqrt_t))
        )

    def term_F() -> float:
        # Knock-out rebate paid at first-touch time. Closed form for
        # E^Q[e^{-r tau} 1{tau <= T}] under the log-spot drift
        # mu_f = r - q - sigma^2/2, with lam_f = sqrt(mu_f^2 + 2 r sigma^2).
        mu_f = r - q - 0.5 * sigma2
        lam_f = math.sqrt(mu_f * mu_f + 2.0 * r * sigma2)
        b = math.log(H / S)  # signed log-barrier distance (<0 down, >0 up)
        if eta_rb > 0.0:  # down
            return rebate * (
                (H / S) ** ((mu_f - lam_f) / sigma2)
                * norm_cdf((b - lam_f * T) / sigma_sqrt_t)
                + (H / S) ** ((mu_f + lam_f) / sigma2)
                * norm_cdf((b + lam_f * T) / sigma_sqrt_t)
            )
        # up
        return rebate * (
            (H / S) ** ((-mu_f + lam_f) / sigma2)
            * norm_cdf((-b - lam_f * T) / sigma_sqrt_t)
            + (H / S) ** ((-mu_f - lam_f) / sigma2)
            * norm_cdf((-b + lam_f * T) / sigma_sqrt_t)
        )

    is_in = barrier_type.endswith("in")
    # Branch split is whether the barrier is below the strike.
    h_below_k = H < K

    if is_call:
        if is_down:  # down call: down-and-in = C (+rebate E), out = A-C (+rebate F)
            if is_in:
                value = term_C() + term_E()
            else:
                value = term_A() - term_C() + term_F()
        else:  # up barrier, S < H
            if not h_below_k:  # K <= H
                if is_in:
                    value = term_B() - term_C() + term_D() + term_E()
                else:
                    value = term_A() - term_B() + term_C() - term_D() + term_F()
            else:  # H < K
                if is_in:
                    value = term_A() + term_E()
                else:
                    value = term_F()
    else:  # put
        if is_down:  # down barrier, H < S (eta = -1)
            if not h_below_k:  # K <= H
                if is_in:
                    value = term_A() + term_E()
                else:
                    # A down-and-out put with strike at/below the barrier can
                    # never pay off: it is pure first-touch rebate.
                    value = term_F()
            else:  # H < K
                if is_in:
                    value = term_B() + term_C() - term_D() + term_E()
                else:
                    value = term_A() - term_B() - term_C() + term_D() + term_F()
        else:  # up barrier, S < H (eta = -1 for put)
            if not h_below_k:  # K <= H
                if is_in:
                    value = term_C() + term_E()
                else:
                    value = term_A() - term_C() + term_F()
            else:  # H < K
                if is_in:
                    value = term_A() - term_B() + term_D() + term_E()
                else:
                    value = term_B() - term_D() + term_F()

    if not math.isfinite(value):
        raise PricingError(
            ErrorCode.NUMERICAL_INSTABILITY,
            "barrier price is not finite",
            {"price": repr(value)},
        )
    # Cancellation of large nearly-equal terms can leave tiny negatives.
    if -1e-8 < value < 0.0:
        value = 0.0
    if value < 0.0:
        raise PricingError(
            ErrorCode.NUMERICAL_INSTABILITY,
            "barrier price is significantly negative (formula breakdown)",
            {"price": repr(value), "barrier_type": barrier_type},
        )
    return value

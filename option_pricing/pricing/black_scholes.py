"""Black-Scholes 欧式期权解析定价与希腊值。

仅依赖标准库 math，正态 CDF 用 math.erf 实现，避免引入 scipy。
"""

from __future__ import annotations

import math

from ..contracts import Greeks, OptionType
from ..errors import NumericalInstabilityError

_SQRT_2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


def _norm_pdf(x: float) -> float:
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def _d1_d2(spot: float, strike: float, maturity: float,
           rate: float, vol: float) -> tuple[float, float]:
    sqrt_t = math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * maturity) / (vol * sqrt_t)
    return d1, d1 - vol * sqrt_t


def _check_finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise NumericalInstabilityError(
            f"Black-Scholes 中间结果 {name} 非有限值",
            detail={"quantity": name},
        )
    return value


def bs_price(option_type: OptionType, spot: float, strike: float,
             maturity: float, rate: float, vol: float) -> float:
    d1, d2 = _d1_d2(spot, strike, maturity, rate, vol)
    disc = math.exp(-rate * maturity)
    if option_type is OptionType.CALL:
        price = spot * _norm_cdf(d1) - strike * disc * _norm_cdf(d2)
    else:
        price = strike * disc * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
    return _check_finite(price, "price")


def bs_greeks(option_type: OptionType, spot: float, strike: float,
              maturity: float, rate: float, vol: float) -> Greeks:
    d1, d2 = _d1_d2(spot, strike, maturity, rate, vol)
    sqrt_t = math.sqrt(maturity)
    disc = math.exp(-rate * maturity)
    pdf_d1 = _norm_pdf(d1)

    gamma = pdf_d1 / (spot * vol * sqrt_t)
    vega = spot * pdf_d1 * sqrt_t  # 每 1.00 波动率；对外按 1% 报告
    if option_type is OptionType.CALL:
        delta = _norm_cdf(d1)
        theta = (-spot * pdf_d1 * vol / (2.0 * sqrt_t)
                 - rate * strike * disc * _norm_cdf(d2))
        rho = strike * maturity * disc * _norm_cdf(d2)
    else:
        delta = _norm_cdf(d1) - 1.0
        theta = (-spot * pdf_d1 * vol / (2.0 * sqrt_t)
                 + rate * strike * disc * _norm_cdf(-d2))
        rho = -strike * maturity * disc * _norm_cdf(-d2)

    return Greeks(
        delta=_check_finite(delta, "delta"),
        gamma=_check_finite(gamma, "gamma"),
        vega=_check_finite(vega / 100.0, "vega"),      # 每 1% 波动率
        theta=_check_finite(theta / 365.0, "theta"),   # 每自然日
        rho=_check_finite(rho / 100.0, "rho"),         # 每 1% 利率
    )

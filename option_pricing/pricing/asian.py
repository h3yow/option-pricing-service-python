"""算术平均亚式期权：确定性 Monte Carlo 定价与有限差分希腊值。

可复现性设计：
- 随机数只来自调用方传入的 seed（numpy Generator），同参数同 seed 结果逐位一致；
- 希腊值用中心有限差分 + 共同随机数（CRN），差分噪声大幅低于独立采样；
- 返回蒙特卡洛标准误，供上层判断数值稳定性，绝不静默返回不可靠结果。
"""

from __future__ import annotations

import math

import numpy as np

from ..contracts import Greeks, OptionType
from ..errors import NumericalInstabilityError

_N_PATHS = 40_000  # 实际路径数（含对偶变量则为 2 倍）


def _simulate_payoffs(option_type: OptionType, spot: float, strike: float,
                      maturity: float, rate: float, vol: float,
                      averaging_points: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_steps = max(averaging_points, 1)
    dt = maturity / n_steps
    # 对偶变量：一半路径用 z，另一半用 -z
    half = _N_PATHS // 2
    z = rng.standard_normal((half, n_steps))
    z = np.concatenate([z, -z], axis=0)

    log_s0 = math.log(spot)
    drift = (rate - 0.5 * vol * vol) * dt
    diff = vol * math.sqrt(dt)
    log_paths = log_s0 + np.cumsum(drift + diff * z, axis=1)
    paths = np.exp(log_paths)
    avg = paths.mean(axis=1)

    if option_type is OptionType.CALL:
        payoff = np.maximum(avg - strike, 0.0)
    else:
        payoff = np.maximum(strike - avg, 0.0)
    return payoff * math.exp(-rate * maturity)


def _check(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise NumericalInstabilityError(
            f"亚式期权蒙特卡洛结果 {name} 非有限值", detail={"quantity": name})
    return value


def asian_price(option_type: OptionType, spot: float, strike: float,
                maturity: float, rate: float, vol: float,
                averaging_points: int, *, seed: int) -> tuple[float, float]:
    """返回 (价格, 蒙特卡洛标准误)。同一组参数+seed 必须完全可复现。"""
    payoffs = _simulate_payoffs(option_type, spot, strike, maturity,
                                rate, vol, averaging_points, seed)
    price = _check(float(payoffs.mean()), "price")
    std_err = _check(float(payoffs.std(ddof=1) / math.sqrt(payoffs.size)),
                     "std_err")
    return price, std_err


def asian_greeks(option_type: OptionType, spot: float, strike: float,
                 maturity: float, rate: float, vol: float,
                 averaging_points: int, *, seed: int) -> Greeks:
    """中心有限差分 + 共同随机数。theta 按日、vega/rho 按 1% 报告。"""
    h_s = max(spot * 1e-4, 1e-6)
    h_v = 1e-4
    h_r = 1e-4
    h_t = min(1.0 / 365.0, maturity / 2.0)

    def p(s: float, v: float, r: float, t: float) -> float:
        price, _ = asian_price(option_type, s, strike, t, r, v,
                               averaging_points, seed=seed)
        return price

    p0 = p(spot, vol, rate, maturity)
    p_up = p(spot + h_s, vol, rate, maturity)
    p_dn = p(spot - h_s, vol, rate, maturity)
    p_vu = p(spot, vol + h_v, rate, maturity)
    p_vd = p(spot, vol - h_v, rate, maturity)
    p_ru = p(spot, vol, rate + h_r, maturity)
    p_rd = p(spot, vol, rate - h_r, maturity)
    p_tu = p(spot, vol, rate, maturity + h_t)
    p_td = p(spot, vol, rate, maturity - h_t)

    return Greeks(
        delta=_check((p_up - p_dn) / (2.0 * h_s), "delta"),
        gamma=_check((p_up - 2.0 * p0 + p_dn) / (h_s * h_s), "gamma"),
        vega=_check((p_vu - p_vd) / (2.0 * h_v) / 100.0, "vega"),
        theta=_check((p_td - p_tu) / (2.0 * h_t) / 365.0, "theta"),
        rho=_check((p_ru - p_rd) / (2.0 * h_r) / 100.0, "rho"),
    )

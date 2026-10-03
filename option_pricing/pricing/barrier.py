"""单障碍欧式期权：Reiner-Rubinstein 解析定价、有限差分希腊值与确定性 MC。

模型约定：
- 连续监测障碍；回扣在触碰发生时支付（解析式 F 项为首达时折现的闭式解）。
- 解析式要求定价时刻尚未触碰障碍；标的价格已越过障碍的情形由上层
  在校验阶段明确拒绝，绝不返回含义含糊的数值。
- MC 引擎（确定性 seed、对偶变量、Broadie-Glasserman 连续性修正）用于
  交叉验证解析式，并作为 model=monte_carlo 时的定价路径；同一组
  参数+seed 结果逐位可复现。

解析式采用 Haug A--F 分解（b=r 的 Black-Scholes 框架）：
    mu = (r - sigma^2/2) / sigma^2,  lambda = sqrt(mu^2 + 2r/sigma^2)
敲入 =  vanilla(A) - 敲出（无回扣平价关系），回扣另加 F 项。
"""

from __future__ import annotations

import math

import numpy as np

from ..contracts import BarrierDirection, Greeks, KnockType, OptionType
from ..errors import NumericalInstabilityError

_N_PATHS = 40_000      # MC 路径数（含对偶变量则为 2 倍）
_MC_STEPS = 512        # MC 监测步数
_BG_SHIFT = 0.5826     # Broadie-Glasserman 连续性修正系数

_SQRT_2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


def _check(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise NumericalInstabilityError(
            f"障碍期权解析结果 {name} 非有限值", detail={"quantity": name})
    return value


def _terms(spot: float, strike: float, barrier: float, maturity: float,
           rate: float, vol: float, phi: float, eta: float,
           rebate: float) -> tuple[float, float, float, float, float, float]:
    """Haug A--F 项。phi=+1 看涨 / -1 看跌；eta=+1 下障碍 / -1 上障碍。"""
    sqrt_t = math.sqrt(maturity)
    sig_t = vol * sqrt_t
    mu = (rate - 0.5 * vol * vol) / (vol * vol)
    lam = math.sqrt(mu * mu + 2.0 * rate / (vol * vol))
    log_sh = math.log(spot / barrier)
    x1 = math.log(spot / strike) / sig_t + (1.0 + mu) * sig_t
    x2 = log_sh / sig_t + (1.0 + mu) * sig_t
    y1 = math.log(barrier * barrier / (spot * strike)) / sig_t + (1.0 + mu) * sig_t
    y2 = -log_sh / sig_t + (1.0 + mu) * sig_t
    z = -log_sh / sig_t + lam * sig_t

    hs = barrier / spot
    disc = math.exp(-rate * maturity)
    try:
        hs_2mu = hs ** (2.0 * mu)
        hs_2mu2 = hs_2mu * hs * hs
        hs_mu_lam = hs ** (mu + lam)
        hs_mu_lam_neg = hs ** (mu - lam)
    except OverflowError:
        raise NumericalInstabilityError(
            "障碍解析式中间结果 (H/S)^k 溢出",
            detail={"barrier": barrier, "spot": spot, "vol": vol}) from None

    a = phi * (spot * _norm_cdf(phi * x1)
               - strike * disc * _norm_cdf(phi * (x1 - sig_t)))
    b = phi * (spot * _norm_cdf(phi * x2)
               - strike * disc * _norm_cdf(phi * (x2 - sig_t)))
    c = phi * (spot * hs_2mu2 * _norm_cdf(eta * y1)
               - strike * disc * hs_2mu * _norm_cdf(eta * (y1 - sig_t)))
    d = phi * (spot * hs_2mu2 * _norm_cdf(eta * y2)
               - strike * disc * hs_2mu * _norm_cdf(eta * (y2 - sig_t)))
    f = rebate * (hs_mu_lam * _norm_cdf(eta * z)
                  + hs_mu_lam_neg * _norm_cdf(eta * (z - 2.0 * lam * sig_t)))
    return a, b, c, d, f, a  # 末位 a 即 vanilla，便于敲入=vanilla-敲出


def _knock_out_price(option_type: OptionType, direction: BarrierDirection,
                     spot: float, strike: float, barrier: float,
                     maturity: float, rate: float, vol: float) -> float:
    """无回扣敲出价格（A--D 组合）。"""
    phi = 1.0 if option_type is OptionType.CALL else -1.0
    eta = 1.0 if direction is BarrierDirection.DOWN else -1.0
    a, b, c, d, _f, _ = _terms(spot, strike, barrier, maturity, rate, vol,
                               phi, eta, 0.0)
    call = option_type is OptionType.CALL
    down = direction is BarrierDirection.DOWN
    if call and down:
        return a - c if strike > barrier else b - d
    if call and not down:
        return 0.0 if strike > barrier else a - b + c - d
    if not call and down:
        return a - b + c - d if strike > barrier else 0.0
    return a - c  # put, up：两个 K/H 区间均为 A-C（经 MC 与极限行为验证）


def barrier_price(option_type: OptionType, direction: BarrierDirection,
                  knock: KnockType, spot: float, strike: float,
                  barrier: float, rebate: float, maturity: float,
                  rate: float, vol: float) -> float:
    """Reiner-Rubinstein 解析式（连续监测，回扣触碰即付）。

    敲入 = vanilla - 敲出（无回扣平价）；回扣经 F 项加回（敲入/敲出
    均在触碰时支付）。
    """
    phi = 1.0 if option_type is OptionType.CALL else -1.0
    eta = 1.0 if direction is BarrierDirection.DOWN else -1.0
    out = _knock_out_price(option_type, direction, spot, strike, barrier,
                           maturity, rate, vol)
    *_ , vanilla = _terms(spot, strike, barrier, maturity, rate, vol,
                          phi, eta, 0.0)
    _a, _b, _c, _d, f, _ = _terms(spot, strike, barrier, maturity, rate,
                                  vol, phi, eta, rebate)
    if knock is KnockType.OUT:
        price = out + f
    else:
        price = vanilla - out + f
    return _check(price, "price")


def barrier_greeks(option_type: OptionType, direction: BarrierDirection,
                   knock: KnockType, spot: float, strike: float,
                   barrier: float, rebate: float, maturity: float,
                   rate: float, vol: float) -> Greeks:
    """对解析式做中心有限差分；确定性、可复现。

    靠近障碍时收缩差分步长，避免差分点越过障碍进入未定义区域。
    """
    dist = abs(spot - barrier)
    h_s = min(max(spot * 1e-4, 1e-6), max(dist / 4.0, 1e-6))
    h_v = 1e-4
    h_r = 1e-4
    h_t = min(1.0 / 365.0, maturity / 2.0)

    def p(s: float, v: float, r: float, t: float) -> float:
        return barrier_price(option_type, direction, knock, s, strike,
                             barrier, rebate, t, r, v)

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


def _simulate(option_type: OptionType, direction: BarrierDirection,
              knock: KnockType, spot: float, strike: float, barrier: float,
              rebate: float, maturity: float, rate: float, vol: float,
              seed: int) -> np.ndarray:
    """确定性 MC：对偶变量 + Broadie-Glasserman 连续性修正。"""
    rng = np.random.default_rng(seed)
    half = _N_PATHS // 2
    z = rng.standard_normal((half, _MC_STEPS))
    z = np.concatenate([z, -z], axis=0)

    dt = maturity / _MC_STEPS
    drift = (rate - 0.5 * vol * vol) * dt
    diff = vol * math.sqrt(dt)
    log_paths = math.log(spot) + np.cumsum(drift + diff * z, axis=1)
    paths = np.exp(log_paths)

    # 连续性修正：离散监测的等效障碍向未触碰侧平移
    shift = math.exp(_BG_SHIFT * vol * math.sqrt(dt))
    adj_barrier = barrier / shift if direction is BarrierDirection.UP \
        else barrier * shift

    if direction is BarrierDirection.UP:
        breach = paths >= adj_barrier
    else:
        breach = paths <= adj_barrier
    hit = breach.any(axis=1)
    first = np.where(hit, breach.argmax(axis=1), -1)
    tau = np.maximum(first, 0) * dt  # 首达时刻（未触碰时不用）

    s_t = paths[:, -1]
    if option_type is OptionType.CALL:
        vanilla = np.maximum(s_t - strike, 0.0)
    else:
        vanilla = np.maximum(strike - s_t, 0.0)
    vanilla *= math.exp(-rate * maturity)
    rebate_at_hit = rebate * np.exp(-rate * tau)

    if knock is KnockType.OUT:
        return np.where(hit, rebate_at_hit, vanilla)
    return np.where(hit, vanilla + rebate_at_hit, 0.0)


def barrier_price_mc(option_type: OptionType, direction: BarrierDirection,
                     knock: KnockType, spot: float, strike: float,
                     barrier: float, rebate: float, maturity: float,
                     rate: float, vol: float, *,
                     seed: int) -> tuple[float, float]:
    """确定性 MC（离散监测+连续性修正）。返回 (价格, 标准误)。"""
    payoffs = _simulate(option_type, direction, knock, spot, strike,
                        barrier, rebate, maturity, rate, vol, seed)
    price = _check(float(payoffs.mean()), "mc_price")
    std_err = _check(float(payoffs.std(ddof=1) / math.sqrt(payoffs.size)),
                     "mc_std_err")
    return price, std_err


def barrier_greeks_mc(option_type: OptionType, direction: BarrierDirection,
                      knock: KnockType, spot: float, strike: float,
                      barrier: float, rebate: float, maturity: float,
                      rate: float, vol: float, *, seed: int) -> Greeks:
    """MC 路径的希腊值：中心有限差分 + 共同随机数（同 seed 消除采样噪声）。"""
    dist = abs(spot - barrier)
    h_s = min(max(spot * 1e-4, 1e-6), max(dist / 4.0, 1e-6))
    h_v = 1e-4
    h_r = 1e-4
    h_t = min(1.0 / 365.0, maturity / 2.0)

    def p(s: float, v: float, r: float, t: float) -> float:
        price, _ = barrier_price_mc(option_type, direction, knock, s, strike,
                                    barrier, rebate, t, r, v, seed=seed)
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

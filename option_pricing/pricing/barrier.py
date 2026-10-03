"""单障碍欧式期权：Reiner-Rubinstein 解析定价与有限差分希腊值。

模型范围（双向守住）：
- 仅适用于 style=BARRIER 的欧式单障碍合约（看涨/看跌 × 向上/向下 ×
  敲入/敲出，共 8 个分支），由 service 层在调用前校验；
- 回扣（rebate）约定为触碰发生时生效、到期支付：敲出或敲入一旦被
  触发，持有方在到期日收到 R，现值为 R·e^{-rT}·P(触碰)；
- 若期初标的价格已越过障碍，直接按"已触碰"短路返回，不套公式。

确定性：纯解析式，无随机数，同参数结果逐位一致。
公式经连续监测（Brownian bridge）蒙特卡洛交叉验证。
"""

from __future__ import annotations

import math

from ..contracts import (BarrierDirection, Greeks, KnockType, OptionType)
from ..errors import NumericalInstabilityError
from .black_scholes import _norm_cdf, bs_price

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _check(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise NumericalInstabilityError(
            f"障碍期权解析中间结果 {name} 非有限值", detail={"quantity": name})
    return value


def _terms(spot: float, strike: float, barrier: float, maturity: float,
           rate: float, vol: float, phi: float, eta: float
           ) -> tuple[float, float, float, float]:
    """Reiner-Rubinstein 的 A/B/C/D 四项（Haug 记号，成本持有率 b=r）。"""
    sqrt_t = math.sqrt(maturity)
    sig_t = vol * sqrt_t
    mu = (rate - 0.5 * vol * vol) / (vol * vol)
    disc = math.exp(-rate * maturity)

    x1 = math.log(spot / strike) / sig_t + (1.0 + mu) * sig_t
    x2 = math.log(spot / barrier) / sig_t + (1.0 + mu) * sig_t
    y1 = math.log(barrier * barrier / (spot * strike)) / sig_t + (1.0 + mu) * sig_t
    y2 = math.log(barrier / spot) / sig_t + (1.0 + mu) * sig_t

    pow_2mu = (barrier / spot) ** (2.0 * mu)
    pow_2mu2 = (barrier / spot) ** (2.0 * (mu + 1.0))

    a = (phi * spot * _norm_cdf(phi * x1)
         - phi * strike * disc * _norm_cdf(phi * x1 - phi * sig_t))
    b = (phi * spot * _norm_cdf(phi * x2)
         - phi * strike * disc * _norm_cdf(phi * x2 - phi * sig_t))
    # 反射项：标的支腿指数 2(μ+1)，行权价支腿指数 2μ（Haug 记号）
    c = (phi * spot * pow_2mu2 * _norm_cdf(eta * y1)
         - phi * strike * disc * pow_2mu
         * _norm_cdf(eta * y1 - eta * sig_t))
    d = (phi * spot * pow_2mu2 * _norm_cdf(eta * y2)
         - phi * strike * disc * pow_2mu
         * _norm_cdf(eta * y2 - eta * sig_t))
    return a, b, c, d


def _no_touch_rebate(rebate: float, spot: float, barrier: float,
                     maturity: float, rate: float, vol: float,
                     eta: float) -> float:
    """到期支付、从未触碰障碍时获得的回扣现值（Haug 的 E 项）。"""
    if rebate == 0.0:
        return 0.0
    sqrt_t = math.sqrt(maturity)
    sig_t = vol * sqrt_t
    mu = (rate - 0.5 * vol * vol) / (vol * vol)
    x2 = math.log(spot / barrier) / sig_t + (1.0 + mu) * sig_t
    y2 = math.log(barrier / spot) / sig_t + (1.0 + mu) * sig_t
    pow_2mu = (barrier / spot) ** (2.0 * mu)
    e = (rebate * math.exp(-rate * maturity)
         * (_norm_cdf(eta * (x2 - sig_t))
            - pow_2mu * _norm_cdf(eta * (y2 - sig_t))))
    return _check(e, "rebate_term")


def _out_value(option_type: OptionType, direction: BarrierDirection,
               spot: float, strike: float, barrier: float, maturity: float,
               rate: float, vol: float) -> float:
    """无回扣敲出价值。phi=+1 看涨 / -1 看跌，eta=+1 向下 / -1 向上。

    分支组合（Haug 记号，已经连续监测蒙特卡洛在 K<>H 两区间、远近障碍
    下交叉验证）：
    - 下敲出看涨 / 上敲出看跌：A−C；
    - 上敲出看涨 / 下敲出看跌：A−B+C−D（在 K 越过 H 使合约必然敲出时
      自然退化为 0）。
    """
    phi = 1.0 if option_type is OptionType.CALL else -1.0
    eta = 1.0 if direction is BarrierDirection.DOWN else -1.0
    a, b, c, d = _terms(spot, strike, barrier, maturity, rate, vol, phi, eta)
    vanilla_side = option_type is OptionType.CALL
    down = direction is BarrierDirection.DOWN
    if vanilla_side == down:  # 下敲出看涨、上敲出看跌
        value = a - c
    else:                     # 上敲出看涨、下敲出看跌
        value = a - b + c - d
    return max(value, 0.0)


def _already_knocked(direction: BarrierDirection, spot: float,
                     barrier: float) -> bool:
    if direction is BarrierDirection.DOWN:
        return spot <= barrier
    return spot >= barrier


def barrier_price(option_type: OptionType, direction: BarrierDirection,
                  knock: KnockType, spot: float, strike: float,
                  barrier: float, maturity: float, rate: float, vol: float,
                  rebate: float = 0.0) -> float:
    """单障碍欧式期权解析价格（回扣在触碰发生时生效、到期支付）。"""
    try:
        return _barrier_price(option_type, direction, knock, spot, strike,
                              barrier, maturity, rate, vol, rebate)
    except OverflowError:
        # 极低波动率下反射因子 (H/S)^{2μ} 溢出：不是正常定价区间
        raise NumericalInstabilityError(
            "障碍期权解析式在该波动率下数值溢出",
            detail={"vol": vol, "barrier": barrier, "spot": spot})


def _barrier_price(option_type: OptionType, direction: BarrierDirection,
                   knock: KnockType, spot: float, strike: float,
                   barrier: float, maturity: float, rate: float, vol: float,
                   rebate: float = 0.0) -> float:
    disc = math.exp(-rate * maturity)
    vanilla = bs_price(option_type, spot, strike, maturity, rate, vol)

    if _already_knocked(direction, spot, barrier):
        # 期初已触碰：敲出只剩回扣，敲入退化为 vanilla + 回扣
        if knock is KnockType.OUT:
            return _check(rebate * disc, "price")
        return _check(vanilla + rebate * disc, "price")

    eta = 1.0 if direction is BarrierDirection.DOWN else -1.0
    out = _out_value(option_type, direction, spot, strike, barrier,
                     maturity, rate, vol)
    # 触碰发生时生效、到期支付的回扣现值 = R·e^{-rT}·P(触碰) = R·e^{-rT} − E
    touch_rebate = rebate * disc - _no_touch_rebate(
        rebate, spot, barrier, maturity, rate, vol, eta)

    if knock is KnockType.OUT:
        price = out + touch_rebate
    else:
        # 敲入的 vanilla 支腿 = vanilla − 敲出（in-out 平价）
        price = (vanilla - out) + touch_rebate
    return _check(max(price, 0.0), "price")


def barrier_greeks(option_type: OptionType, direction: BarrierDirection,
                   knock: KnockType, spot: float, strike: float,
                   barrier: float, maturity: float, rate: float, vol: float,
                   rebate: float = 0.0) -> Greeks:
    """中心有限差分希腊值（解析式求导，确定性、无 MC 噪声）。

    theta 按自然日、vega/rho 按 1% 报告，与 BS 输出口径一致。
    """
    h_s = max(spot * 1e-4, 1e-6)
    h_v = 1e-4
    h_r = 1e-4
    h_t = min(1.0 / 365.0, maturity / 2.0)

    def p(s: float, v: float, r: float, t: float) -> float:
        return barrier_price(option_type, direction, knock, s, strike,
                             barrier, t, r, v, rebate)

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

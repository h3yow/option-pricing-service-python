"""单障碍合约单测：解析式与 MC 互验、平价关系、模型守卫、入口一致性。

日志打印每条用例的输入参数与判定依据。
"""

import logging

import pytest

from option_pricing.calibration import CalibrationSession
from option_pricing.contracts import (
    BarrierDirection,
    Contract,
    ExerciseStyle,
    KnockType,
    MarketParams,
    Model,
    OptionType,
)
from option_pricing.errors import ModelNotApplicableError, ValidationError
from option_pricing.pricing.barrier import (
    barrier_greeks,
    barrier_price,
    barrier_price_mc,
)
from option_pricing.pricing.black_scholes import bs_price
from option_pricing.service import PricingService

log = logging.getLogger(__name__)

S, K, T, R, V = 100.0, 100.0, 1.0, 0.05, 0.25
M = MarketParams(spot=S, volatility=V, risk_free_rate=R)
H_UP, H_DOWN = 105.0, 95.0
REBATE = 3.0


def _mk(option_type, direction, knock, strike=K, barrier=None, rebate=REBATE):
    if barrier is None:
        barrier = H_UP if direction is BarrierDirection.UP else H_DOWN
    return Contract(f"B-{option_type.value}-{direction.value}-{knock.value}",
                    option_type, ExerciseStyle.BARRIER_SINGLE, strike, T,
                    barrier=barrier, barrier_direction=direction, knock=knock,
                    rebate=rebate)


def _analytic(c, spot=S, vol=V):
    return barrier_price(c.option_type, c.barrier_direction, c.knock, spot,
                         c.strike, c.barrier, c.rebate, c.maturity_years, R,
                         vol)


def test_analytic_matches_mc_all_branches():
    """8 个分支（看涨/看跌 × 上/下 × 敲入/敲出）与确定性 MC 交叉验证。"""
    for ot in (OptionType.CALL, OptionType.PUT):
        for direction in (BarrierDirection.UP, BarrierDirection.DOWN):
            for knock in (KnockType.IN, KnockType.OUT):
                c = _mk(ot, direction, knock)
                analytic = _analytic(c)
                mc, se = barrier_price_mc(
                    ot, direction, knock, S, c.strike, c.barrier, c.rebate,
                    T, R, V, seed=7)
                tol = 4.0 * se + 0.005 * max(analytic, 1.0)
                log.info("%s %s %s K=%s H=%s R=%s: 解析=%.5f MC=%.5f "
                         "(se=%.5f, 容差=%.5f); 判定依据: 连续监测解析式与"
                         "离散监测+连续性修正 MC 应在容差内一致",
                         ot.value, direction.value, knock.value, c.strike,
                         c.barrier, c.rebate, analytic, mc, se, tol)
                assert abs(analytic - mc) <= tol


def test_in_out_parity_equals_vanilla():
    """无回扣时 敲入+敲出=vanilla（对每个分支精确成立）。"""
    for ot in (OptionType.CALL, OptionType.PUT):
        for direction in (BarrierDirection.UP, BarrierDirection.DOWN):
            ci = _mk(ot, direction, KnockType.IN, rebate=0.0)
            co = _mk(ot, direction, KnockType.OUT, rebate=0.0)
            vanilla = bs_price(ot, S, K, T, R, V)
            total = _analytic(ci) + _analytic(co)
            log.info("%s %s: in+out=%.10f vanilla=%.10f; 判定依据: 平价关系"
                     "逐位成立", ot.value, direction.value, total, vanilla)
            assert total == pytest.approx(vanilla, abs=1e-9)


def test_knock_out_degenerate_when_strike_beyond_barrier():
    """上看涨敲出 K>H 恒为 0（+回扣项）；下看跌敲出 K<H 恒为 0。"""
    c = _mk(OptionType.CALL, BarrierDirection.UP, KnockType.OUT,
            strike=110.0, barrier=105.0, rebate=0.0)
    p = _mk(OptionType.PUT, BarrierDirection.DOWN, KnockType.OUT,
            strike=90.0, barrier=95.0, rebate=0.0)
    log.info("uo-call K=110>H=105 价格=%.6f; do-put K=90<H=95 价格=%.6f;"
             "判定依据: 要实值必先触碰障碍, 无回扣时应为 0",
             _analytic(c), _analytic(p))
    assert _analytic(c) == 0.0 and _analytic(p) == 0.0


def test_rebate_floor_for_knock_out():
    """带回扣敲出合约价格不低于回扣折现的下界（深度虚值时趋近回扣项）。"""
    far = _mk(OptionType.CALL, BarrierDirection.UP, KnockType.OUT,
              strike=200.0, barrier=105.0, rebate=REBATE)
    price = _analytic(far)
    log.info("uo-call K=200 H=105 R=3: 价格=%.5f; 判定依据: 几乎必然触碰,"
             "价格应接近回扣首达折现值且为正", price)
    assert 0.0 < price < REBATE


def test_mc_deterministic_same_seed():
    c = _mk(OptionType.CALL, BarrierDirection.UP, KnockType.OUT)
    p1, e1 = barrier_price_mc(OptionType.CALL, BarrierDirection.UP,
                              KnockType.OUT, S, K, c.barrier, REBATE, T, R, V,
                              seed=11)
    p2, e2 = barrier_price_mc(OptionType.CALL, BarrierDirection.UP,
                              KnockType.OUT, S, K, c.barrier, REBATE, T, R, V,
                              seed=11)
    log.info("同 seed 两次 MC: %.12f / %.12f; 判定依据: 逐位一致", p1, p2)
    assert p1 == p2 and e1 == e2


def test_greeks_sane_and_finite():
    for ot in (OptionType.CALL, OptionType.PUT):
        for direction in (BarrierDirection.UP, BarrierDirection.DOWN):
            for knock in (KnockType.IN, KnockType.OUT):
                c = _mk(ot, direction, knock)
                g = barrier_greeks(ot, direction, knock, S, K, c.barrier,
                                   REBATE, T, R, V)
                log.info("%s %s %s 希腊值 %s; 判定依据: 全部有限",
                         ot.value, direction.value, knock.value, g)
                assert all(abs(x) < 1e6 for x in
                           (g.delta, g.gamma, g.vega, g.theta, g.rho))


# ---------------------------------------------------------------- 服务层

_BARRIER_CONTRACTS = [
    _mk(OptionType.CALL, BarrierDirection.UP, KnockType.OUT),
    _mk(OptionType.PUT, BarrierDirection.DOWN, KnockType.IN),
    _mk(OptionType.CALL, BarrierDirection.DOWN, KnockType.OUT),
    _mk(OptionType.PUT, BarrierDirection.UP, KnockType.IN),
]

# 各合约在基准波动率 0.25 下的解析价作为会话报价（校准应回到 0.25 附近）
_QUOTES = {c.contract_id: _analytic(c) for c in _BARRIER_CONTRACTS}


@pytest.fixture()
def service():
    return PricingService(CalibrationSession(dict(_QUOTES), market=M))


def test_model_guard_both_directions(service):
    eu = Contract("EU", OptionType.CALL, ExerciseStyle.EUROPEAN, K, T)
    az = Contract("AZ", OptionType.CALL, ExerciseStyle.ASIAN_ARITHMETIC, K, T)
    ba = _mk(OptionType.CALL, BarrierDirection.UP, KnockType.OUT)
    cases = [
        (eu, Model.MONTE_CARLO, "欧式只接受 black_scholes"),
        (eu, Model.BARRIER_ANALYTIC, "欧式不接受障碍解析式"),
        (az, Model.BLACK_SCHOLES, "亚式不接受 BS"),
        (az, Model.BARRIER_ANALYTIC, "亚式不接受障碍解析式"),
        (ba, Model.BLACK_SCHOLES, "障碍合约不接受 BS"),
    ]
    for contract, model, why in cases:
        log.info("模型守卫(%s): style=%s model=%s; 判定依据: 出数前必须"
                 "拒绝 model_not_applicable", why, contract.style.value,
                 model.value)
        with pytest.raises(ModelNotApplicableError):
            service.price(contract, M, model=model)


def test_barrier_accepts_analytic_and_mc(service):
    c = _mk(OptionType.CALL, BarrierDirection.UP, KnockType.OUT)
    r1 = service.price(c, M)
    r2 = service.price(c, M, model=Model.BARRIER_ANALYTIC)
    r3 = service.price(c, M, model=Model.MONTE_CARLO)
    log.info("默认模型=%s 显式解析=%s MC=%.5f (解析=%.5f); 判定依据: 默认即"
             "解析式, MC 与解析接近且可复现", r1.model.value, r2.model.value,
             r3.price, r1.price)
    assert r1.model is Model.BARRIER_ANALYTIC and r1 == r2
    assert r3.model is Model.MONTE_CARLO
    assert abs(r3.price - r1.price) < 0.02 * r1.price + 4 * r3.diagnostics[
        "mc_std_err"]


def test_breached_barrier_rejected_before_pricing(service):
    c = _mk(OptionType.CALL, BarrierDirection.UP, KnockType.OUT)
    for spot in (105.0, 106.0):
        market = MarketParams(spot=spot, volatility=V, risk_free_rate=R)
        log.info("spot=%s 已触及上障碍 105; 判定依据: -> validation_error, "
                 "绝不返回含义含糊的数值", spot)
        with pytest.raises(ValidationError):
            service.price(c, market)


def test_single_equals_batch_and_reproducible(service):
    singles = [service.price(c, M) for c in _BARRIER_CONTRACTS]
    batch = service.price_batch([(c, M) for c in _BARRIER_CONTRACTS])
    log.info("单笔=%s 批量=%s; 判定依据: 同一计算路径逐位一致",
             [r.price for r in singles], [r.price for r in batch])
    assert batch == singles
    again = service.price(_BARRIER_CONTRACTS[0], M)
    assert again == singles[0]

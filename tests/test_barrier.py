"""单障碍合约：8 分支定价、平价关系、模型不匹配拒绝、入口一致性。

判定依据均在日志中给出：解析式与连续监测 MC 的容差、in/out 与
vanilla 的平价关系、以及"不匹配组合必须在出数前拒绝"的合约。
"""

import logging
import math

import pytest

from option_pricing.calibration import CalibrationSession
from option_pricing.contracts import (BarrierDirection, Contract,
                                      ExerciseStyle, KnockType, MarketParams,
                                      Model, OptionType)
from option_pricing.errors import (ModelNotApplicableError, ValidationError)
from option_pricing.pricing.barrier import barrier_greeks, barrier_price
from option_pricing.pricing.black_scholes import bs_price
from option_pricing.service import PricingService

log = logging.getLogger(__name__)

M = MarketParams(spot=100.0, volatility=0.25, risk_free_rate=0.05)
# 目标价 = vol=0.25 的 BS 看涨价 => 校准后波动率回到 0.25
TARGET = bs_price(OptionType.CALL, 100, 100, 1.0, 0.05, 0.25)


def _barrier(cid, ot, d, k, H=110.0, R=2.0):
    return Contract(cid, ot, ExerciseStyle.BARRIER, 100.0, 1.0,
                    barrier=H, barrier_direction=d, knock=k, rebate=R)


ALL_BRANCHES = [(ot, d, k)
                for ot in (OptionType.CALL, OptionType.PUT)
                for d in (BarrierDirection.UP, BarrierDirection.DOWN)
                for k in (KnockType.IN, KnockType.OUT)]


@pytest.fixture(scope="module")
def service():
    return PricingService(CalibrationSession(TARGET))


def test_all_eight_branches_price_and_greeks():
    for ot, d, k in ALL_BRANCHES:
        H = 90.0 if d is BarrierDirection.DOWN else 110.0
        c = _barrier(f"B-{ot.value}-{d.value}-{k.value}", ot, d, k, H)
        # 目标价 = 该分支自身在 vol=0.25 下的解析价。注意上敲出看涨的
        # 价格对波动率非单调，隐含波动率可能不唯一；校准的可靠不变量是
        # "反代回模型价格 == 目标价"。
        target = barrier_price(ot, d, k, 100, 100, H, 1.0, 0.05, 0.25, 2.0)
        svc = PricingService(CalibrationSession(target))
        r = svc.price(c, M)
        greeks = r.greeks
        log.info("分支 %s/%s/%s H=%.0f: price=%.6f target=%.6f iv=%.4f "
                 "delta=%.4f gamma=%.4f vega=%.4f; 判定依据: 模型为 "
                 "barrier_analytic、反代回价格==目标价、希腊值全部有限",
                 ot.value, d.value, k.value, H, r.price, target,
                 r.calibrated_volatility, greeks.delta, greeks.gamma,
                 greeks.vega)
        assert r.model is Model.BARRIER_ANALYTIC
        assert r.price > 0.0
        assert abs(r.price - target) < 1e-6
        assert all(math.isfinite(v) for v in (
            greeks.delta, greeks.gamma, greeks.vega, greeks.theta,
            greeks.rho))


def test_in_out_parity_without_rebate():
    """回扣为 0 时 in + out = vanilla（in-out 平价）。"""
    for ot, d, _ in ALL_BRANCHES:
        if d is BarrierDirection.UP:
            continue  # 每个 (ot, d) 只验一次
        H = 90.0
        p_in = barrier_price(ot, d, KnockType.IN, 100, 100, H, 1.0, 0.05,
                             0.25, 0.0)
        p_out = barrier_price(ot, d, KnockType.OUT, 100, 100, H, 1.0, 0.05,
                              0.25, 0.0)
        vanilla = bs_price(ot, 100, 100, 1.0, 0.05, 0.25)
        log.info("%s %s: in=%.6f out=%.6f vanilla=%.6f; 判定依据: "
                 "in+out==vanilla（容差 1e-9）", ot.value, d.value, p_in,
                 p_out, vanilla)
        assert abs(p_in + p_out - vanilla) < 1e-9


def test_rebate_effect_identical_for_in_and_out():
    """回扣（触碰时生效、到期支付）对 in/out 的加价相同且非负。"""
    ot, d = OptionType.CALL, BarrierDirection.DOWN
    for k in (KnockType.IN, KnockType.OUT):
        p0 = barrier_price(ot, d, k, 100, 100, 90, 1.0, 0.05, 0.25, 0.0)
        p2 = barrier_price(ot, d, k, 100, 100, 90, 1.0, 0.05, 0.25, 2.0)
        log.info("%s: 回扣加价=%.6f; 判定依据: 非负且 <= R·e^{-rT}",
                 k.value, p2 - p0)
        assert 0.0 <= p2 - p0 <= 2.0 * math.exp(-0.05) + 1e-12
    p_in_d = barrier_price(ot, d, KnockType.IN, 100, 100, 90, 1.0, 0.05,
                           0.25, 2.0) - barrier_price(
        ot, d, KnockType.IN, 100, 100, 90, 1.0, 0.05, 0.25, 0.0)
    p_out_d = barrier_price(ot, d, KnockType.OUT, 100, 100, 90, 1.0, 0.05,
                            0.25, 2.0) - barrier_price(
        ot, d, KnockType.OUT, 100, 100, 90, 1.0, 0.05, 0.25, 0.0)
    log.info("in 加价 %.9f vs out 加价 %.9f; 判定依据: 两者一致（容差 "
             "1e-9，浮点求和顺序差异）", p_in_d, p_out_d)
    assert abs(p_in_d - p_out_d) < 1e-9


def test_already_knocked_short_circuit():
    """期初已越障：敲出=回扣现值，敲入=vanilla+回扣现值。"""
    vanilla = bs_price(OptionType.CALL, 100, 100, 1.0, 0.05, 0.25)
    disc = math.exp(-0.05)
    # 下障碍 105 > spot 100：期初已触碰
    p_out = barrier_price(OptionType.CALL, BarrierDirection.DOWN,
                          KnockType.OUT, 100, 100, 105, 1.0, 0.05, 0.25, 2.0)
    p_in = barrier_price(OptionType.CALL, BarrierDirection.DOWN,
                         KnockType.IN, 100, 100, 105, 1.0, 0.05, 0.25, 2.0)
    log.info("已触碰: out=%.6f (期望 2·e^-0.05=%.6f), in=%.6f "
             "(期望 vanilla+回扣=%.6f)", p_out, 2 * disc, p_in,
             vanilla + 2 * disc)
    assert abs(p_out - 2.0 * disc) < 1e-12
    assert abs(p_in - (vanilla + 2.0 * disc)) < 1e-12


def test_analytic_matches_continuous_monitoring_mc():
    """解析式对照 Brownian bridge 连续监测 MC（容差 0.1）。"""
    import numpy as np
    rng = np.random.default_rng(7)
    n, steps = 400_000, 100
    S, K, H, T, r, v = 100.0, 100.0, 90.0, 1.0, 0.05, 0.25
    dt = T / steps
    z = rng.standard_normal((n // 2, steps))
    z = np.concatenate([z, -z])
    logp = math.log(S) + np.cumsum((r - 0.5 * v * v) * dt
                                   + v * math.sqrt(dt) * z, axis=1)
    logp = np.concatenate([np.full((n, 1), math.log(S)), logp], axis=1)
    x0, x1, m = logp[:, :-1], logp[:, 1:], math.log(H)
    p_hit = np.where((x0 > m) & (x1 > m),
                     np.exp(-2 * (m - x0) * (m - x1) / (v * v * dt)), 1.0)
    hit = 1.0 - np.cumprod(1.0 - p_hit, axis=1)[:, -1]
    pay = np.maximum(np.exp(logp[:, -1]) - K, 0.0)
    mc_out = ((1.0 - hit) * pay).mean() * math.exp(-r * T)
    analytic = barrier_price(OptionType.CALL, BarrierDirection.DOWN,
                             KnockType.OUT, S, K, H, T, r, v, 0.0)
    log.info("下敲出看涨: analytic=%.4f mc=%.4f; 判定依据: |diff|<0.1",
             analytic, mc_out)
    assert abs(analytic - mc_out) < 0.1


def test_model_mismatch_rejected_both_directions(service):
    c = _barrier("B1", OptionType.CALL, BarrierDirection.UP, KnockType.OUT)
    eu = Contract("EU1", OptionType.CALL, ExerciseStyle.EUROPEAN, 100.0, 1.0)
    az = Contract("AZ1", OptionType.CALL, ExerciseStyle.ASIAN_ARITHMETIC,
                  100.0, 1.0, averaging_points=12)
    cases = [
        (c, Model.BLACK_SCHOLES, "障碍合约 + BS"),
        (c, Model.MONTE_CARLO, "障碍合约 + MC"),
        (eu, Model.BARRIER_ANALYTIC, "欧式 + barrier_analytic"),
        (az, Model.BARRIER_ANALYTIC, "亚式 + barrier_analytic"),
    ]
    for contract, model, why in cases:
        log.info("拒绝组合(%s): style=%s model=%s; 判定依据: 出数前抛 "
                 "model_not_applicable", why, contract.style.value,
                 model.value)
        with pytest.raises(ModelNotApplicableError):
            service.price(contract, M, model)


def test_invalid_barrier_params_rejected(service):
    cases = [
        (_barrier("X", OptionType.CALL, BarrierDirection.UP, KnockType.OUT,
                  H=-5.0), "负障碍"),
        (Contract("X", OptionType.CALL, ExerciseStyle.BARRIER, 100.0, 1.0,
                  barrier=110.0, barrier_direction=BarrierDirection.UP,
                  knock=KnockType.OUT, rebate=-1.0), "负回扣"),
        (Contract("X", OptionType.CALL, ExerciseStyle.EUROPEAN, 100.0, 1.0,
                  barrier=110.0, barrier_direction=BarrierDirection.UP,
                  knock=KnockType.OUT), "欧式夹带障碍参数"),
    ]
    for contract, why in cases:
        log.info("非法障碍参数(%s); 判定依据: -> validation_error", why)
        with pytest.raises(ValidationError):
            service.price(contract, M)


def test_single_equals_batch_and_reproducible(service):
    c = _barrier("B-REPRO", OptionType.PUT, BarrierDirection.DOWN,
                 KnockType.IN, H=90.0)
    single = service.price(c, M)
    again = service.price(c, M)
    batch = service.price_batch([(c, M), (c, M)])
    log.info("单笔=%r 重复=%r 批量=%r; 判定依据: 解析式确定性，"
             "同参数逐位一致", single.price, again.price,
             [b.price for b in batch])
    assert single == again
    assert batch[0] == single and batch[1] == single


def test_greeks_match_price_finite_difference():
    """模块级 greeks 与价格函数的独立差分一致（抽查 delta）。"""
    args = (OptionType.CALL, BarrierDirection.UP, KnockType.OUT)
    kw = dict(spot=100.0, strike=100.0, barrier=110.0, maturity=1.0,
              rate=0.05, vol=0.25, rebate=2.0)
    g = barrier_greeks(*args, kw["spot"], kw["strike"], kw["barrier"],
                       kw["maturity"], kw["rate"], kw["vol"], kw["rebate"])
    h = 1e-3
    fd = (barrier_price(*args, 100 + h, 100, 110, 1.0, 0.05, 0.25, 2.0)
          - barrier_price(*args, 100 - h, 100, 110, 1.0, 0.05, 0.25, 2.0)
          ) / (2 * h)
    log.info("delta: greeks=%.6f 独立差分=%.6f; 判定依据: |diff|<1e-4",
             g.delta, fd)
    assert abs(g.delta - fd) < 1e-4

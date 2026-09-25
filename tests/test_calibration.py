"""校准会话单测：重校准、并发复用、错误分类、缓存有界。"""

import logging
import threading

import pytest

from option_pricing.calibration import CalibrationSession
from option_pricing.contracts import (Contract, ExerciseStyle, MarketParams,
                                      OptionType)
from option_pricing.errors import (NoConvergenceError, NoSolutionError,
                                   ValidationError)
from option_pricing.pricing.black_scholes import bs_price

log = logging.getLogger(__name__)

M = MarketParams(spot=100.0, volatility=0.2, risk_free_rate=0.05)
C = Contract("C1", OptionType.CALL, ExerciseStyle.EUROPEAN, 100.0, 1.0)
TARGET = bs_price(OptionType.CALL, 100, 100, 1.0, 0.05, 0.25)


def test_round_trip_implied_vol():
    iv = CalibrationSession(TARGET).calibrated_volatility(C, M)
    log.info("目标价 %.6f 反解 iv=%.8f; 判定依据: 应回到生成目标价的 0.25",
             TARGET, iv)
    assert iv == pytest.approx(0.25, abs=1e-8)


def test_param_change_forces_recalibration():
    s = CalibrationSession(TARGET)
    iv1 = s.calibrated_volatility(C, M)
    m2 = MarketParams(spot=105.0, volatility=0.2, risk_free_rate=0.05)
    iv2 = s.calibrated_volatility(C, m2)
    log.info("spot 100->105: iv %.6f -> %.6f; 判定依据: 键变化必须重算",
             iv1, iv2)
    assert iv2 != iv1 and s.size == 2


def test_no_solution_when_target_out_of_range():
    log.info("目标价 1.0 低于看涨下限 S-Ke^-rT=%.4f; 判定依据: 应报 no_solution",
             100 - 100 * 0.951229)
    with pytest.raises(NoSolutionError):
        CalibrationSession(1.0).calibrated_volatility(C, M)


def test_no_convergence_with_tiny_budget():
    s = CalibrationSession(TARGET, max_iter=1)
    with pytest.raises(NoConvergenceError):
        s.calibrated_volatility(C, M)
    log.info("max_iter=1 判定依据: 应报 no_convergence 而非底层异常")


def test_invalid_target_rejected():
    with pytest.raises(ValidationError):
        CalibrationSession(-3.0)


def test_concurrent_single_flight_consistent():
    s = CalibrationSession(TARGET)
    results, errors = [], []

    def work():
        try:
            results.append(s.calibrated_volatility(C, M))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log.info("32 线程并发: 结果数=%d 唯一值=%s 缓存大小=%d; 判定依据: "
             "全部一致且只缓存 1 项", len(results), set(results), s.size)
    assert not errors and len(results) == 32
    assert len(set(results)) == 1 and s.size == 1


def test_cache_bounded():
    s = CalibrationSession(TARGET, max_entries=4)
    for i in range(10):
        s.calibrated_volatility(
            Contract(f"C{i}", OptionType.CALL, ExerciseStyle.EUROPEAN,
                     100.0, 1.0), M)
    log.info("校准 10 个合约后缓存大小=%d; 判定依据: 不超过上限 4", s.size)
    assert s.size == 4

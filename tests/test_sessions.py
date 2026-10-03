"""校准会话：生命周期、更新后重新校准、并发隔离、容量上限、批量复用。

一致性判定依据：
- 更新 = 版本 +1 且校准缓存整体重建，后续价格必须反映新报价；
- 并发读写期间，每条结果整体属于某一版（session_version 与价格同源）；
- 不同会话的校准缓存互不复用。
"""

import logging
import threading

import pytest

from option_pricing.contracts import (Contract, ExerciseStyle, MarketParams,
                                      OptionType)
from option_pricing.errors import (CapacityExceededError, QuoteNotFoundError,
                                   SessionNotFoundError, ValidationError)
from option_pricing.pricing.black_scholes import bs_price
from option_pricing.service import PricingService
from option_pricing.sessions import SessionManager

log = logging.getLogger(__name__)

M = MarketParams(spot=100.0, volatility=0.25, risk_free_rate=0.05)
EU = Contract("EU-CALL-100", OptionType.CALL, ExerciseStyle.EUROPEAN,
              100.0, 1.0)
Q1 = bs_price(OptionType.CALL, 100, 100, 1.0, 0.05, 0.25)  # iv -> 0.25
Q2 = bs_price(OptionType.CALL, 100, 100, 1.0, 0.05, 0.35)  # iv -> 0.35


@pytest.fixture()
def mgr():
    return SessionManager()


def _service():
    from option_pricing.calibration import CalibrationSession
    return PricingService(CalibrationSession(Q1))


def test_lifecycle(mgr):
    snap = mgr.create(M, {"EU-CALL-100": Q1}, session_id="client-a")
    log.info("创建会话 client-a v%d; 判定依据: 可查、可更新、可删",
             snap.version)
    assert mgr.get("client-a").version == 1
    assert mgr.get("client-a").quotes["EU-CALL-100"] == Q1

    snap2 = mgr.update("client-a", quotes={"EU-CALL-100": Q2})
    assert snap2.version == 2
    assert mgr.get("client-a").version == 2

    mgr.delete("client-a")
    with pytest.raises(SessionNotFoundError):
        mgr.get("client-a")
    with pytest.raises(SessionNotFoundError):
        mgr.delete("client-a")
    log.info("删除后查询/再删均 session_not_found; 判定依据: 可区分错误")


def test_duplicate_id_and_empty_quotes_rejected(mgr):
    mgr.create(M, {"EU-CALL-100": Q1}, session_id="s1")
    with pytest.raises(ValidationError):
        mgr.create(M, {"EU-CALL-100": Q1}, session_id="s1")
    with pytest.raises(ValidationError):
        mgr.create(M, {}, session_id="s2")
    with pytest.raises(ValidationError):
        mgr.create(M, {"EU-CALL-100": -1.0}, session_id="s3")
    log.info("重复 ID / 空报价 / 负报价均 validation_error; 判定依据: "
             "非法输入在创建期拒绝且不产生状态")
    assert mgr.count() == 1


def test_update_forces_recalibration(mgr):
    svc = _service()
    mgr.create(M, {"EU-CALL-100": Q1}, session_id="s1")
    r1 = svc.price_in_session(mgr.get("s1"), EU)
    mgr.update("s1", quotes={"EU-CALL-100": Q2})
    r2 = svc.price_in_session(mgr.get("s1"), EU)
    log.info("更新前 iv=%.4f v%d; 更新后 iv=%.4f v%d; 判定依据: 版本 +1 且 "
             "校准结果必须基于新报价（0.25 -> 0.35）",
             r1.calibrated_volatility, r1.session_version,
             r2.calibrated_volatility, r2.session_version)
    assert r1.session_version == 1 and r2.session_version == 2
    assert abs(r1.calibrated_volatility - 0.25) < 1e-6
    assert abs(r2.calibrated_volatility - 0.35) < 1e-6


def test_concurrent_read_during_update_is_version_consistent(mgr):
    """更新进行中并发读取：每条结果必须整体属于旧版或新版。"""
    svc = _service()
    mgr.create(M, {"EU-CALL-100": Q1}, session_id="s1")
    stop = threading.Event()
    results = []
    errors = []

    def reader():
        while not stop.is_set():
            try:
                r = svc.price_in_session(mgr.get("s1"), EU)
                results.append(r)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for i in range(50):
        # v1=Q1(初始)；第 i 次更新后版本为 i+2，令奇数版=Q1、偶数版=Q2，
        # 使读者能用版本奇偶校验 iv 来源
        q = Q2 if i % 2 == 0 else Q1
        mgr.update("s1", quotes={"EU-CALL-100": q})
    stop.set()
    for t in threads:
        t.join()

    log.info("50 次更新 × 4 读线程共 %d 条结果, %d 个错误; 判定依据: 每条"
             "结果的 (version, iv) 必须同源——v 奇数->iv0.25(Q1), v 偶数->"
             "iv0.35(Q2)，不允许混合", len(results), len(errors))
    assert not errors and results
    for r in results:
        expected = 0.25 if r.session_version % 2 == 1 else 0.35
        assert r.session_id == "s1"
        assert abs(r.calibrated_volatility - expected) < 1e-6


def test_sessions_are_isolated(mgr):
    svc = _service()
    mgr.create(M, {"EU-CALL-100": Q1}, session_id="iso-a")
    mgr.create(M, {"EU-CALL-100": Q2}, session_id="iso-b")
    ra = svc.price_in_session(mgr.get("iso-a"), EU)
    rb = svc.price_in_session(mgr.get("iso-b"), EU)
    log.info("同一合约: 会话A iv=%.4f, 会话B iv=%.4f; 判定依据: 各自校准"
             "各自报价，互不复用", ra.calibrated_volatility,
             rb.calibrated_volatility)
    assert abs(ra.calibrated_volatility - 0.25) < 1e-6
    assert abs(rb.calibrated_volatility - 0.35) < 1e-6
    assert mgr.get("iso-a").calibration is not mgr.get("iso-b").calibration


def test_missing_quote_rejected(mgr):
    svc = _service()
    mgr.create(M, {"EU-CALL-100": Q1}, session_id="s1")
    other = Contract("EU-PUT-100", OptionType.PUT, ExerciseStyle.EUROPEAN,
                     100.0, 1.0)
    with pytest.raises(QuoteNotFoundError):
        svc.price_in_session(mgr.get("s1"), other)
    log.info("会话无 EU-PUT-100 报价; 判定依据: -> quote_not_found，"
             "而不是拿别的会话或内置目标价凑数")


def test_session_capacity_limit():
    mgr = SessionManager(max_sessions=4)
    for i in range(4):
        mgr.create(M, {"EU-CALL-100": Q1}, session_id=f"s{i}")
    with pytest.raises(CapacityExceededError) as exc:
        mgr.create(M, {"EU-CALL-100": Q1}, session_id="s4")
    log.info("上限 4 个会话, 第 5 个被拒: %s; 判定依据: capacity_exceeded "
             "且 resource=sessions", exc.value.detail)
    assert exc.value.detail["resource"] == "sessions"
    mgr.delete("s0")
    mgr.create(M, {"EU-CALL-100": Q1}, session_id="s4")  # 删除后释放名额
    assert mgr.count() == 4


def test_quotes_capacity_limit():
    mgr = SessionManager()
    quotes = {f"C{i}": Q1 for i in range(65)}
    with pytest.raises(CapacityExceededError) as exc:
        mgr.create(M, quotes)
    log.info("65 条报价超过 64 上限; 判定依据: capacity_exceeded 且 "
             "resource=quotes_per_session")
    assert exc.value.detail["resource"] == "quotes_per_session"


def test_batch_size_limit(mgr):
    svc = _service()
    mgr.create(M, {"EU-CALL-100": Q1}, session_id="s1")
    with pytest.raises(CapacityExceededError) as exc:
        svc.price_batch_in_session(mgr.get("s1"), [EU] * 65)
    log.info("批量 65 超过 64 上限; 判定依据: capacity_exceeded 且 "
             "resource=batch_size")
    assert exc.value.detail["resource"] == "batch_size"


def test_batch_reuses_session_calibration(mgr):
    svc = _service()
    mgr.create(M, {"EU-CALL-100": Q1}, session_id="s1")
    snap = mgr.get("s1")
    results = svc.price_batch_in_session(snap, [EU] * 16)
    size = mgr.get("s1").calibration.size
    log.info("16 笔同一合约批量后缓存条目=%d; 判定依据: 同参数只校准一次"
             "（缓存/single-flight 复用），且全部结果一致", size)
    assert size == 1
    assert all(r == results[0] for r in results)


def test_concurrent_sessions_match_serial(mgr):
    """跨会话并发与串行结果一致，资源占用有界。"""
    svc = _service()
    for i in range(8):
        mgr.create(M, {"EU-CALL-100": Q1 if i % 2 == 0 else Q2},
                   session_id=f"c{i}")
    serial = [svc.price_in_session(mgr.get(f"c{i}"), EU) for i in range(8)]
    concurrent = [None] * 8

    def work(i):
        snap = mgr.get(f"c{i}")
        concurrent[i] = svc.price_batch_in_session(snap, [EU] * 4)[0]

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log.info("8 会话并发 vs 串行; 判定依据: 逐位一致; 总会话数 %d 有界",
             mgr.count())
    assert concurrent == serial
    total_cache = sum(mgr.get(f"c{i}").calibration.size for i in range(8))
    assert total_cache <= 8 * 128

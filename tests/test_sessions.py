"""校准会话单测：生命周期、更新后重校准、并发隔离、容量上限。

日志打印每条用例的输入参数与判定依据。
"""

import logging
import threading

import pytest

from option_pricing.calibration import CalibrationSession
from option_pricing.contracts import Contract, ExerciseStyle, MarketParams, OptionType
from option_pricing.errors import (
    CapacityExceededError,
    NoSolutionError,
    QuoteMissingError,
    SessionNotFoundError,
    ValidationError,
)
from option_pricing.pricing.black_scholes import bs_price
from option_pricing.service import PriceJob, PricingService
from option_pricing.sessions import SessionManager

log = logging.getLogger(__name__)

M = MarketParams(spot=100.0, volatility=0.25, risk_free_rate=0.05)
C1 = Contract("C1", OptionType.CALL, ExerciseStyle.EUROPEAN, 100.0, 1.0)
C2 = Contract("C2", OptionType.PUT, ExerciseStyle.EUROPEAN, 100.0, 1.0)
REG = {"C1": C1, "C2": C2}
Q1 = bs_price(OptionType.CALL, 100, 100, 1.0, 0.05, 0.25)   # -> iv 0.25
Q2 = bs_price(OptionType.PUT, 100, 100, 1.0, 0.05, 0.30)    # -> iv 0.30
Q1B = bs_price(OptionType.CALL, 100, 100, 1.0, 0.05, 0.40)  # -> iv 0.40


@pytest.fixture()
def manager():
    return SessionManager(REG.get)


@pytest.fixture()
def service(manager):
    return PricingService(CalibrationSession(Q1), manager)


# ---------------------------------------------------------------- 生命周期

def test_create_get_list_delete(manager):
    info = manager.create("clientA", "XYZ", {"C1": Q1, "C2": Q2}, M)
    log.info("创建会话 %s quotes=%s version=%d; 判定依据: 可查询且内容一致",
             info.session_id, info.quotes, info.version)
    got = manager.info(info.session_id)
    assert got.quotes == {"C1": Q1, "C2": Q2} and got.version == 0
    assert got.market == M and got.client == "clientA"
    assert any(s.session_id == info.session_id for s in manager.list())

    manager.delete(info.session_id)
    with pytest.raises(SessionNotFoundError):
        manager.info(info.session_id)
    log.info("删除后查询; 判定依据: -> unknown_session")


def test_create_with_unreachable_quote_rejected(manager):
    log.info("报价 1.0 低于看涨下限; 判定依据: 创建即拒绝, 不留半成品会话")
    with pytest.raises(NoSolutionError):
        manager.create("clientA", "XYZ", {"C1": 1.0}, M)
    assert manager.list() == []


def test_create_with_unknown_contract_quote_rejected(manager):
    with pytest.raises(ValidationError):
        manager.create("clientA", "XYZ", {"NOPE": 5.0}, M)
    assert manager.list() == []


def test_delete_unknown_session(manager):
    with pytest.raises(SessionNotFoundError):
        manager.delete("S9999")


# ---------------------------------------------------------------- 更新与重校准

def test_quote_update_forces_recalibration(manager, service):
    info = manager.create("clientA", "XYZ", {"C1": Q1, "C2": Q2}, M)
    r0 = service.price_in_session(C1, info.session_id)
    assert r0.session_version == 0
    assert r0.calibrated_volatility == pytest.approx(0.25, abs=1e-8)

    new_version = manager.update(info.session_id,
                                 quotes={"C1": Q1B, "C2": Q2}).version
    r1 = service.price_in_session(C1, info.session_id)
    log.info("报价 %.4f->%.4f: iv %.6f->%.6f version %d->%d; 判定依据: "
             "更新后必须基于新报价重校准", Q1, Q1B, r0.calibrated_volatility,
             r1.calibrated_volatility, r0.session_version, r1.session_version)
    assert new_version == 1 and r1.session_version == 1
    assert r1.calibrated_volatility == pytest.approx(0.40, abs=1e-8)
    assert r1.price != r0.price


def test_market_update_forces_recalibration(manager, service):
    info = manager.create("clientA", "XYZ", {"C1": Q1}, M)
    r0 = service.price_in_session(C1, info.session_id)
    m2 = MarketParams(spot=105.0, volatility=0.25, risk_free_rate=0.05)
    manager.update(info.session_id, market=m2)
    r1 = service.price_in_session(C1, info.session_id)
    log.info("spot 100->105: iv %.6f->%.6f; 判定依据: 行情更新强制重校准",
             r0.calibrated_volatility, r1.calibrated_volatility)
    assert r1.session_version == 1
    assert r1.calibrated_volatility != r0.calibrated_volatility


def test_unreachable_update_rejected_atomically(manager, service):
    info = manager.create("clientA", "XYZ", {"C1": Q1, "C2": Q2}, M)
    r0 = service.price_in_session(C1, info.session_id)
    with pytest.raises(NoSolutionError):
        manager.update(info.session_id, quotes={"C1": 1.0, "C2": Q2})
    r1 = service.price_in_session(C1, info.session_id)
    log.info("不可达更新被拒后 version=%d iv=%.6f; 判定依据: 状态保持旧值",
             r1.session_version, r1.calibrated_volatility)
    assert r1.session_version == 0 and r1 == r0


def test_quote_missing_rejected(manager, service):
    info = manager.create("clientA", "XYZ", {"C1": Q1}, M)
    log.info("会话只报了 C1, 请求 C2; 判定依据: -> quote_missing")
    with pytest.raises(QuoteMissingError):
        service.price_in_session(C2, info.session_id)


def test_sessions_isolated(manager, service):
    a = manager.create("clientA", "XYZ", {"C1": Q1}, M)
    b = manager.create("clientB", "XYZ", {"C1": Q1B}, M)
    ra = service.price_in_session(C1, a.session_id)
    rb = service.price_in_session(C1, b.session_id)
    log.info("会话 A iv=%.6f 会话 B iv=%.6f; 判定依据: 校准结果互不复用",
             ra.calibrated_volatility, rb.calibrated_volatility)
    assert ra.calibrated_volatility == pytest.approx(0.25, abs=1e-8)
    assert rb.calibrated_volatility == pytest.approx(0.40, abs=1e-8)
    # 更新 A 不影响 B
    manager.update(a.session_id, quotes={"C1": Q1B})
    rb2 = service.price_in_session(C1, b.session_id)
    assert rb2 == rb and rb2.session_version == 0


# ---------------------------------------------------------------- 并发

def test_concurrent_readers_see_old_or_new_never_mixed(manager, service):
    """更新进行中并发读取：每条结果要么整体旧(v0)、要么整体新(v1)。"""
    info = manager.create("clientA", "XYZ", {"C1": Q1}, M)
    expect_v0 = service.price_in_session(C1, info.session_id)
    manager.update(info.session_id, quotes={"C1": Q1B})
    expect_v1 = service.price_in_session(C1, info.session_id)
    manager.update(info.session_id, quotes={"C1": Q1})  # 回到 v0 内容
    assert expect_v0.session_version == 0 and expect_v1.session_version == 1

    stop = threading.Event()
    results, errors = [], []

    def reader():
        while not stop.is_set():
            try:
                results.append(service.price_in_session(C1, info.session_id))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    def writer():
        for _ in range(20):
            manager.update(info.session_id, quotes={"C1": Q1B})
            manager.update(info.session_id, quotes={"C1": Q1})

    threads = ([threading.Thread(target=reader) for _ in range(8)]
               + [threading.Thread(target=writer)])
    for t in threads:
        t.start()
    for t in threads[8:]:
        t.join()
    stop.set()
    for t in threads[:8]:
        t.join()

    by_version: dict[int, list] = {}
    for r in results:
        by_version.setdefault(r.session_version, []).append(r)
    log.info("并发读取 %d 条, 版本分布=%s, 错误=%d; 判定依据: 每个版本只有"
             "唯一结果(内容与版本同源), 无两半混合",
             len(results), {v: len(s) for v, s in by_version.items()},
             len(errors))
    assert not errors
    # 内容必须与对应版本的期望逐位一致（同版本内全部相同）
    for version, group in by_version.items():
        expect = expect_v0 if version % 2 == 0 else expect_v1
        for got in group:
            assert got.price == expect.price
            assert got.calibrated_volatility == expect.calibrated_volatility


def test_concurrent_sessions_consistent_with_serial(manager, service):
    """跨会话并发定价与串行逐条执行结果逐位一致。"""
    a = manager.create("clientA", "XYZ", {"C1": Q1, "C2": Q2}, M)
    b = manager.create("clientB", "XYZ", {"C1": Q1B, "C2": Q2}, M)
    jobs = [(C1, a.session_id), (C2, a.session_id),
            (C1, b.session_id), (C2, b.session_id)]
    serial = [service.price_in_session(c, sid) for c, sid in jobs]

    results: list = [None] * 16

    def work(i):
        c, sid = jobs[i % 4]
        results[i] = service.price_in_session(c, sid)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log.info("串行=%s; 并发 16 线程; 判定依据: 逐位一致",
             [(r.price, r.session_version) for r in serial])
    for i, r in enumerate(results):
        assert r == serial[i % 4]


# ---------------------------------------------------------------- 容量

def test_session_count_capped():
    manager = SessionManager(REG.get, max_sessions=4)
    for i in range(4):
        manager.create(f"client{i}", "XYZ", {"C1": Q1}, M)
    log.info("已建 4 个会话(上限 4); 判定依据: 第 5 个 -> capacity_exceeded"
             "(resource=sessions)")
    with pytest.raises(CapacityExceededError) as exc_info:
        manager.create("client5", "XYZ", {"C1": Q1}, M)
    assert exc_info.value.detail["resource"] == "sessions"
    # 删除一个后可再建
    manager.delete(manager.list()[0].session_id)
    manager.create("client5", "XYZ", {"C1": Q1}, M)


def test_session_cache_bounded():
    s = CalibrationSession({"C1": Q1}, market=M, max_entries=4,
                           contract_resolver=REG.get)
    for i in range(10):
        s.calibrated_volatility(
            Contract("C1", OptionType.CALL, ExerciseStyle.EUROPEAN,
                     100.0 + i, 1.0), M)
    log.info("10 个不同行权价校准后缓存=%d; 判定依据: 不超过上限 4", s.size)
    assert s.size == 4


def test_batch_reuses_session_calibration(manager, service):
    info = manager.create("clientA", "XYZ", {"C1": Q1}, M)
    session = manager.get(info.session_id)
    jobs = [PriceJob(contract=C1, session_id=info.session_id)] * 8
    service.price_each(jobs, M)
    size_after = session.size
    log.info("同会话批量 8 笔同一合约后缓存=%d; 判定依据: 只校准一次",
             size_after)
    assert size_after == 1


def test_batch_size_capped(manager, service):
    info = manager.create("clientA", "XYZ", {"C1": Q1}, M)
    jobs = [PriceJob(contract=C1, session_id=info.session_id)] * 65
    log.info("批量 65 笔(上限 64); 判定依据: -> capacity_exceeded"
             "(resource=batch_size)")
    with pytest.raises(CapacityExceededError) as exc_info:
        service.price_each(jobs, M)
    assert exc_info.value.detail["resource"] == "batch_size"


def test_concurrency_slots_capped(manager):
    service = PricingService(CalibrationSession(Q1), manager,
                             max_in_flight=2)
    # 占满 2 个槽位后再请求 -> capacity_exceeded(resource=concurrency)
    assert service._slots.acquire(blocking=False)
    assert service._slots.acquire(blocking=False)
    log.info("并发槽位 2/2 占满; 判定依据: 第 3 个请求 -> capacity_exceeded"
             "(resource=concurrency)")
    with pytest.raises(CapacityExceededError) as exc_info:
        service.price(C1, M)
    assert exc_info.value.detail["resource"] == "concurrency"
    service._slots.release()
    service._slots.release()
    # 释放后恢复
    service.price(C1, M)

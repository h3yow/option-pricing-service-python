import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pytest

from app.calibration import CalibrationService
from app.enums import OptionType
from app.errors import ErrorCode, PricingError
from app.models import MarketData, MarketQuote
from app.pricing.black_scholes import bs_price

AS_OF = date(2026, 9, 25)
EXPIRY = date(2027, 9, 25)


def quotes_for(S, sig=0.3, r=0.05, q=0.02):
    data = [(95, OptionType.CALL), (100, OptionType.CALL), (105, OptionType.PUT)]
    return [
        MarketQuote(k, EXPIRY, ot, round(bs_price(ot.value, S, k, 1.0, r, q, sig), 6))
        for k, ot in data
    ]


@pytest.fixture()
def service():
    svc = CalibrationService(max_sessions=4, max_workers=8)
    yield svc
    svc.shutdown()


def test_recovers_known_sigma(service, caplog):
    caplog.set_level(logging.INFO)
    s = service.create_session(MarketData(100, 0.05, 0.02), quotes_for(100), AS_OF)
    res = s.get_calibration()
    logging.getLogger("tests").info(
        "calibrate S=100 quotes=%s -> sigma=%.6f sse=%.2e",
        [(q.strike, q.premium) for q in s.quotes], res.sigma, res.sse,
    )
    assert res.sigma == pytest.approx(0.3, abs=1e-4)


def test_change_spot_invalidates_and_recalibrates(service):
    s = service.create_session(MarketData(100, 0.05, 0.02), quotes_for(100), AS_OF)
    r1 = s.get_calibration()
    s.update_market(spot=110.0)
    assert s.version == 2
    # old quotes are inconsistent with new spot -> distinct error
    with pytest.raises(PricingError) as exc:
        s.get_calibration()
    assert exc.value.code is ErrorCode.CALIBRATION_NO_SOLUTION
    # replace quotes -> recalibration succeeds
    s.update_quotes(quotes_for(110))
    r2 = s.get_calibration()
    assert r2 is not r1 and r2.version == 3
    assert r2.sigma == pytest.approx(0.3, abs=1e-4)


def test_change_rate_invalidates(service):
    s = service.create_session(MarketData(100, 0.05, 0.02), quotes_for(100), AS_OF)
    s.get_calibration()
    s.update_market(risk_free_rate=0.07)
    assert s.get_calibration() is not None


def test_single_flight_under_concurrency(service):
    s = service.create_session(MarketData(100, 0.05, 0.02), quotes_for(100), AS_OF)
    barrier = threading.Barrier(24)
    results: list = []
    lock = threading.Lock()

    def worker():
        barrier.wait()
        r = s.get_calibration()
        with lock:
            results.append(r)

    threads = [threading.Thread(target=worker) for _ in range(24)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 24
    assert all(r is results[0] for r in results)


def test_sessions_isolated_under_concurrency(service):
    sessions = [
        service.create_session(
            MarketData(100.0, 0.05, 0.02), quotes_for(100, sig=0.2 + 0.02 * i), AS_OF
        )
        for i in range(4)
    ]
    out: dict[int, object] = {}
    lock = threading.Lock()

    def work(i, sess):
        r = sess.get_calibration()
        with lock:
            out[i] = r.sigma

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda p: work(*p), enumerate(sessions)))
    for i, sess in enumerate(sessions):
        assert out[i] == pytest.approx(0.2 + 0.02 * i, abs=1e-4)


def test_inconsistent_quotes_did_not_converge(service):
    bad = [
        MarketQuote(100, EXPIRY, OptionType.CALL,
                    round(bs_price("call", 100, 100, 1, 0.05, 0.02, 0.15), 6)),
        MarketQuote(100, EXPIRY, OptionType.PUT,
                    round(bs_price("put", 100, 100, 1, 0.05, 0.02, 0.80), 6)),
    ]
    s = service.create_session(MarketData(100, 0.05, 0.02), bad, AS_OF)
    with pytest.raises(PricingError) as exc:
        s.get_calibration()
    assert exc.value.code is ErrorCode.CALIBRATION_DID_NOT_CONVERGE


def test_impossible_quote_no_solution(service):
    s = service.create_session(
        MarketData(100, 0.05, 0.02),
        [MarketQuote(90, EXPIRY, OptionType.CALL, 0.01)], AS_OF,
    )
    with pytest.raises(PricingError) as exc:
        s.get_calibration()
    assert exc.value.code is ErrorCode.CALIBRATION_NO_SOLUTION


def test_empty_quotes_rejected(service):
    with pytest.raises(PricingError) as exc:
        service.create_session(MarketData(100, 0.05, 0.02), [], AS_OF)
    assert exc.value.code is ErrorCode.NO_QUOTES


def test_session_not_found(service):
    with pytest.raises(PricingError) as exc:
        service.get_session("nope")
    assert exc.value.code is ErrorCode.SESSION_NOT_FOUND


def test_session_cap_enforced(service):
    for _ in range(4):
        service.create_session(
            MarketData(100, 0.05, 0.02), quotes_for(100), AS_OF
        )
    with pytest.raises(PricingError) as exc:
        service.create_session(
            MarketData(100, 0.05, 0.02), quotes_for(100), AS_OF
        )
    assert exc.value.code is ErrorCode.INVALID_PARAMETER

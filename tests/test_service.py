import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pytest

from app.calibration import CalibrationService
from app.enums import BarrierType, ModelType, OptionStyle, OptionType
from app.models import (
    ContractSpec,
    MarketData,
    MarketQuote,
    PriceItem,
)
from app.pricing.black_scholes import bs_price
from app.service import PricingService

AS_OF = date(2026, 9, 25)
EXPIRY = date(2027, 9, 25)


@pytest.fixture()
def services():
    calibration = CalibrationService(max_sessions=16, max_workers=8)
    pricing = PricingService(calibration)
    yield pricing, calibration
    calibration.shutdown()


def _euro(cid, k=100.0, sig=0.25):
    return PriceItem(
        contract=ContractSpec(
            contract_id=cid, style=OptionStyle.EUROPEAN,
            option_type=OptionType.CALL, strike=k, expiry=EXPIRY,
        ),
        model=ModelType.BLACK_SCHOLES, sigma_override=sig,
    )


def _barrier(cid, sig=0.25):
    return PriceItem(
        contract=ContractSpec(
            contract_id=cid, style=OptionStyle.BARRIER,
            option_type=OptionType.CALL, strike=100.0, expiry=EXPIRY,
            barrier=120.0, barrier_type=BarrierType.UP_AND_OUT, rebate=2.0,
        ),
        model=ModelType.REINER_RUBINSTEIN, sigma_override=sig,
    )


def test_single_and_batch_bit_identical(services):
    pricing, _ = services
    items = [_euro("a", 95), _euro("b", 105), _barrier("c")]
    singles = [pricing.price_one(it, None) for it in items]
    batch = pricing.price_batch(items, None)
    for s, b in zip(singles, batch):
        assert s == b  # frozen dataclass equality, field by field


def test_batch_matches_serial_under_concurrency(services, caplog):
    pricing, _ = services
    items = [_euro(f"c{i}", 90 + i * 2, 0.2 + 0.01 * i) for i in range(12)]
    items.append(_barrier("b"))
    serial = [pricing.price_one(it, None) for it in items]

    caplog.set_level(logging.INFO)
    logging.getLogger("tests").info(
        "concurrency consistency: %d contracts, %d workers",
        len(items), 8,
    )

    outputs = []
    lock = threading.Lock()

    def run():
        res = pricing.price_batch(items, None)
        with lock:
            outputs.append(res)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: run(), range(10)))

    assert len(outputs) == 10
    for batch in outputs:
        assert batch == serial


def test_batch_reuses_single_calibration(services):
    pricing, calibration = services
    premium = round(bs_price("call", 100, 100, 1, 0.05, 0.02, 0.28), 6)
    session = calibration.create_session(
        MarketData(100, 0.05, 0.02),
        [MarketQuote(100, EXPIRY, OptionType.CALL, premium)], AS_OF,
    )
    items = [
        PriceItem(
            contract=ContractSpec(
                contract_id=f"s{i}", style=OptionStyle.EUROPEAN,
                option_type=OptionType.CALL, strike=100.0, expiry=EXPIRY,
            ),
            model=ModelType.BLACK_SCHOLES,
        )
        for i in range(20)
    ]
    results = pricing.price_batch(items, session.session_id)
    assert all(r.calibrated for r in results)
    # every result references the same fitted sigma and the session ran once
    sigmas = {r.sigma_used for r in results}
    assert sigmas == {session.get_calibration().sigma}


def test_parameter_change_forces_recalibration(services):
    pricing, calibration = services
    q0 = round(bs_price("call", 100, 100, 1, 0.05, 0.02, 0.3), 6)
    session = calibration.create_session(
        MarketData(100, 0.05, 0.02),
        [MarketQuote(100, EXPIRY, OptionType.CALL, q0)], AS_OF,
    )
    first = session.get_calibration()
    # new spot with consistent new quotes
    q1 = round(bs_price("call", 105, 100, 1, 0.05, 0.02, 0.33), 6)
    session.update_market(spot=105.0)
    session.update_quotes(
        [MarketQuote(100, EXPIRY, OptionType.CALL, q1)]
    )
    second = session.get_calibration()
    assert second is not first
    assert second.sigma != first.sigma
    assert second.version == 3

import logging
from datetime import date

import pytest

from app.enums import BarrierType, ModelType, OptionStyle, OptionType
from app.errors import ErrorCode, PricingError
from app.models import MarketData, OptionContract
from app.validation import (
    tenor_years,
    validate_contract,
    validate_market,
    validate_model,
    validate_volatility,
)

AS_OF = date(2026, 9, 25)


def _european(k=100.0):
    return OptionContract("c", OptionStyle.EUROPEAN, OptionType.CALL, k,
                          date(2027, 9, 25))


def test_market_bounds():
    for bad in (0.0, -1.0, 1e12, float("nan")):
        with pytest.raises(PricingError) as exc:
            validate_market(MarketData(bad, 0.05, 0.0))
        assert exc.value.code is ErrorCode.INVALID_PARAMETER


def test_volatility_bounds():
    for bad in (0.0, -0.2, 6.0):
        with pytest.raises(PricingError) as exc:
            validate_volatility(bad)
        assert exc.value.code is ErrorCode.INVALID_PARAMETER


def test_expired_contract():
    with pytest.raises(PricingError) as exc:
        tenor_years(date(2026, 9, 25), AS_OF)
    assert exc.value.code is ErrorCode.CONTRACT_EXPIRED
    with pytest.raises(PricingError) as exc:
        tenor_years(date(2025, 1, 1), AS_OF)
    assert exc.value.code is ErrorCode.CONTRACT_EXPIRED


def test_tenor_daycount(caplog):
    caplog.set_level(logging.INFO)
    t = tenor_years(date(2027, 3, 26), AS_OF)  # 182 days
    logging.getLogger("tests").info("tenor days=182 -> years=%s", t)
    assert t == pytest.approx(182 / 365.0, abs=1e-12)


def test_barrier_contract_requires_barrier_fields():
    c = OptionContract("b", OptionStyle.BARRIER, OptionType.CALL, 100.0,
                       date(2027, 1, 1))
    with pytest.raises(PricingError) as exc:
        validate_contract(c, AS_OF)
    assert exc.value.code is ErrorCode.INVALID_CONTRACT_SPEC


def test_european_rejects_barrier_fields():
    c = OptionContract("e", OptionStyle.EUROPEAN, OptionType.CALL, 100.0,
                       date(2027, 1, 1), barrier=120.0,
                       barrier_type=BarrierType.UP_AND_OUT)
    with pytest.raises(PricingError) as exc:
        validate_contract(c, AS_OF)
    assert exc.value.code is ErrorCode.INVALID_CONTRACT_SPEC


def test_model_applicability():
    c = _european()
    with pytest.raises(PricingError) as exc:
        validate_model(c, ModelType.REINER_RUBINSTEIN, 100.0)
    assert exc.value.code is ErrorCode.MODEL_NOT_APPLICABLE

    b = OptionContract("b", OptionStyle.BARRIER, OptionType.PUT, 100.0,
                       date(2027, 1, 1), barrier=90.0,
                       barrier_type=BarrierType.DOWN_AND_OUT)
    with pytest.raises(PricingError) as exc:
        validate_model(b, ModelType.BLACK_SCHOLES, 100.0)
    assert exc.value.code is ErrorCode.MODEL_NOT_APPLICABLE
    # already breached
    with pytest.raises(PricingError) as exc:
        validate_model(b, ModelType.REINER_RUBINSTEIN, 80.0)
    assert exc.value.code is ErrorCode.MODEL_NOT_APPLICABLE
    # live barrier is fine
    validate_model(b, ModelType.REINER_RUBINSTEIN, 100.0)

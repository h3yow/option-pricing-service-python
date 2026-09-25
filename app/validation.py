"""Centralised validation for market parameters, contracts and models.

Every rejection maps to a distinguishable ErrorCode so callers never receive
a raw ValueError or a silent NaN.
"""
from __future__ import annotations

from datetime import date

from app.enums import ModelType, OptionStyle
from app.errors import ErrorCode, PricingError
from app.models import MarketData, OptionContract

DAY_COUNT_DENOMINATOR = 365.0

# Bounds chosen to be far outside any realistic market regime while still
# producing numerically well-defined model values.
_SPOT_MIN, _SPOT_MAX = 1e-6, 1.0e9
_STRIKE_MIN, _STRIKE_MAX = 1e-6, 1.0e9
_RATE_MIN, _RATE_MAX = -0.99, 5.0
_YIELD_MIN, _YIELD_MAX = -0.99, 5.0
_VOL_MIN_EXCLUSIVE = 0.0
_VOL_MAX = 5.0
_TENOR_MIN_EXCLUSIVE = 0.0
_TENOR_MAX = 50.0
_BARRIER_MIN = 1e-6
_REBATE_MIN = 0.0


def validate_market(market: MarketData) -> None:
    """Validate spot/rates. Volatility is validated where it is supplied."""

    def check(name: str, value: float, lo: float, hi: float) -> None:
        if not isinstance(value, (int, float)) or not _finite(value):
            raise PricingError(
                ErrorCode.INVALID_PARAMETER,
                f"market parameter {name} must be a finite number",
                {"parameter": name, "value": repr(value)},
            )
        if not (lo <= value <= hi):
            raise PricingError(
                ErrorCode.INVALID_PARAMETER,
                f"market parameter {name}={value} outside allowed range "
                f"[{lo}, {hi}]",
                {"parameter": name, "value": value, "min": lo, "max": hi},
            )

    check("spot", market.spot, _SPOT_MIN, _SPOT_MAX)
    check("risk_free_rate", market.risk_free_rate, _RATE_MIN, _RATE_MAX)
    check("dividend_yield", market.dividend_yield, _YIELD_MIN, _YIELD_MAX)


def validate_volatility(sigma: float) -> None:
    if not _finite(sigma):
        raise PricingError(
            ErrorCode.INVALID_PARAMETER,
            "volatility must be a finite number",
            {"value": repr(sigma)},
        )
    if not (_VOL_MIN_EXCLUSIVE < sigma <= _VOL_MAX):
        raise PricingError(
            ErrorCode.INVALID_PARAMETER,
            f"volatility sigma={sigma} outside allowed range "
            f"({_VOL_MIN_EXCLUSIVE}, {_VOL_MAX}]",
            {"parameter": "sigma", "value": sigma,
             "min_exclusive": _VOL_MIN_EXCLUSIVE, "max": _VOL_MAX},
        )


def validate_contract(contract: OptionContract, as_of: date) -> None:
    """Validate structural consistency of a contract specification."""
    if not isinstance(contract.strike, (int, float)) or not _finite(contract.strike):
        raise PricingError(
            ErrorCode.INVALID_CONTRACT_SPEC,
            "strike must be a finite number",
            {"contract_id": contract.contract_id, "strike": repr(contract.strike)},
        )
    if not (_STRIKE_MIN <= contract.strike <= _STRIKE_MAX):
        raise PricingError(
            ErrorCode.INVALID_CONTRACT_SPEC,
            f"strike={contract.strike} outside allowed range",
            {"contract_id": contract.contract_id, "strike": contract.strike},
        )
    if not _finite(contract.rebate) or contract.rebate < _REBATE_MIN:
        raise PricingError(
            ErrorCode.INVALID_CONTRACT_SPEC,
            "rebate must be a non-negative finite number",
            {"contract_id": contract.contract_id, "rebate": repr(contract.rebate)},
        )

    if contract.style is OptionStyle.BARRIER:
        if contract.barrier is None or contract.barrier_type is None:
            raise PricingError(
                ErrorCode.INVALID_CONTRACT_SPEC,
                "barrier contract requires both barrier level and barrier type",
                {"contract_id": contract.contract_id},
            )
        if not _finite(contract.barrier) or contract.barrier <= _BARRIER_MIN:
            raise PricingError(
                ErrorCode.INVALID_CONTRACT_SPEC,
                f"barrier level must be > {_BARRIER_MIN}",
                {"contract_id": contract.contract_id,
                 "barrier": repr(contract.barrier)},
            )
    else:
        if contract.barrier is not None or contract.barrier_type is not None:
            raise PricingError(
                ErrorCode.INVALID_CONTRACT_SPEC,
                "barrier level/type must not be set on a European contract",
                {"contract_id": contract.contract_id},
            )


def validate_model(
    contract: OptionContract,
    model: ModelType,
    spot: float,
) -> None:
    """Ensure the selected model is applicable to the contract and its current
    barrier configuration."""
    if contract.style is OptionStyle.EUROPEAN:
        if model is not ModelType.BLACK_SCHOLES:
            raise PricingError(
                ErrorCode.MODEL_NOT_APPLICABLE,
                f"model {model.value} cannot price a European option; "
                f"only {ModelType.BLACK_SCHOLES.value} is supported",
                {"model": model.value, "style": contract.style.value},
            )
    elif contract.style is OptionStyle.BARRIER:
        if model is not ModelType.REINER_RUBINSTEIN:
            raise PricingError(
                ErrorCode.MODEL_NOT_APPLICABLE,
                f"model {model.value} cannot price a barrier option; "
                f"only {ModelType.REINER_RUBINSTEIN.value} is supported",
                {"model": model.value, "style": contract.style.value},
            )
        # The analytic barrier formula assumes the barrier has not already
        # been breached at inception.
        assert contract.barrier is not None
        is_down = contract.barrier_type.value.startswith("down")
        breached = spot <= contract.barrier if is_down else spot >= contract.barrier
        if breached:
            raise PricingError(
                ErrorCode.MODEL_NOT_APPLICABLE,
                "barrier already breached at the valuation spot: the "
                "standard Reiner-Rubinstein formula assumes a live barrier",
                {"spot": spot, "barrier": contract.barrier,
                 "barrier_type": contract.barrier_type.value},
            )


def tenor_years(expiry: date, as_of: date) -> float:
    """ACT/365 fixed day count; raises CONTRACT_EXPIRED when non-positive."""
    days = (expiry - as_of).days
    t = days / DAY_COUNT_DENOMINATOR
    if days <= 0:
        raise PricingError(
            ErrorCode.CONTRACT_EXPIRED,
            f"contract already expired: expiry={expiry.isoformat()} "
            f"is not after as_of={as_of.isoformat()}",
            {"expiry": expiry.isoformat(), "as_of": as_of.isoformat()},
        )
    if t > _TENOR_MAX:
        raise PricingError(
            ErrorCode.INVALID_PARAMETER,
            f"tenor {t:.2f} years exceeds maximum {_TENOR_MAX}",
            {"tenor_years": t, "max": _TENOR_MAX},
        )
    return t


def validate_tenor_range(t: float) -> None:
    if not _finite(t) or not (_TENOR_MIN_EXCLUSIVE < t <= _TENOR_MAX):
        raise PricingError(
            ErrorCode.INVALID_PARAMETER,
            f"tenor must be in ({_TENOR_MIN_EXCLUSIVE}, {_TENOR_MAX}] years",
            {"tenor_years": t},
        )


def _finite(value: float) -> bool:
    return isinstance(value, (int, float)) and value == value and value not in (
        float("inf"),
        float("-inf"),
    )

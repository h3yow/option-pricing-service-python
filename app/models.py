"""Domain types and API request/response schemas."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel, Field

from app.enums import BarrierType, ModelType, OptionStyle, OptionType


# ---------------------------------------------------------------------------
# Domain value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketData:
    spot: float
    risk_free_rate: float
    dividend_yield: float


@dataclass(frozen=True)
class OptionContract:
    contract_id: str
    style: OptionStyle
    option_type: OptionType
    strike: float
    expiry: date
    barrier: float | None = None
    barrier_type: BarrierType | None = None
    rebate: float = 0.0


@dataclass(frozen=True)
class MarketQuote:
    strike: float
    expiry: date
    option_type: OptionType
    premium: float


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


@dataclass(frozen=True)
class ValuationResult:
    contract_id: str
    price: float
    greeks: Greeks
    model: ModelType
    sigma_used: float
    as_of: date
    tenor_years: float
    calibrated: bool


# ---------------------------------------------------------------------------
# API schemas
# ---------------------------------------------------------------------------


class ContractSpec(BaseModel):
    contract_id: str = ""
    style: OptionStyle
    option_type: OptionType
    strike: float
    expiry: date
    barrier: float | None = None
    barrier_type: BarrierType | None = None
    rebate: float = 0.0


class QuoteSpec(BaseModel):
    strike: float
    expiry: date
    option_type: OptionType
    premium: float


class MarketSpec(BaseModel):
    spot: float
    risk_free_rate: float
    dividend_yield: float = 0.0


class CreateSessionRequest(BaseModel):
    market: MarketSpec
    quotes: list[QuoteSpec]


class UpdateMarketRequest(BaseModel):
    spot: float | None = None
    risk_free_rate: float | None = None
    dividend_yield: float | None = None


class UpdateQuotesRequest(BaseModel):
    quotes: list[QuoteSpec]


class PriceItem(BaseModel):
    contract: ContractSpec
    model: ModelType
    as_of: date | None = None
    sigma_override: float | None = Field(default=None, gt=0.0)


class PriceRequest(BaseModel):
    session_id: str | None = None
    items: list[PriceItem]


class GreeksOut(BaseModel):
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class ValuationOut(BaseModel):
    contract_id: str
    price: float
    greeks: GreeksOut
    model: ModelType
    sigma_used: float
    as_of: date
    tenor_years: float
    calibrated: bool


class ErrorOut(BaseModel):
    error_code: str
    message: str
    detail: dict = Field(default_factory=dict)

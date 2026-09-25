"""FastAPI wiring: HTTP routes and exception translation.

Domain errors (:class:`PricingError`) are rendered with a stable
machine-readable ``error_code``; any other exception is caught at the boundary
and reported as a generic INTERNAL_ERROR so numerical-library tracebacks never
leak to clients.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.calibration import CalibrationService
from app.enums import ModelType, OptionStyle
from app.errors import ErrorCode, PricingError
from app.models import (
    CreateSessionRequest,
    GreeksOut,
    MarketData,
    MarketQuote,
    PriceRequest,
    UpdateMarketRequest,
    UpdateQuotesRequest,
    ValuationOut,
)
from app.service import DEFAULT_AS_OF, PricingService

logger = logging.getLogger("option_pricing")


class StatusOut(BaseModel):
    status: str
    default_as_of: str
    supported_styles: list[str]
    supported_models: list[str]


class SessionOut(BaseModel):
    session_id: str
    version: int
    calibrated_sigma: float | None
    quote_count: int


class BatchEntry(BaseModel):
    ok: bool
    result: ValuationOut | None = None
    error_code: str | None = None
    message: str | None = None
    detail: dict | None = None


class BatchResponse(BaseModel):
    count: int
    results: list[BatchEntry]


def _serialize(result) -> ValuationOut:
    return ValuationOut(
        contract_id=result.contract_id,
        price=result.price,
        greeks=GreeksOut(**result.greeks.__dict__),
        model=result.model,
        sigma_used=result.sigma_used,
        as_of=result.as_of,
        tenor_years=result.tenor_years,
        calibrated=result.calibrated,
    )


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Iterator[None]:
        calibration = CalibrationService(max_sessions=256, max_workers=8)
        app.state.calibration = calibration
        app.state.pricing = PricingService(calibration)
        logger.info("pricing service started (as_of=%s)", DEFAULT_AS_OF)
        try:
            yield
        finally:
            calibration.shutdown()
            logger.info("pricing service stopped")

    app = FastAPI(title="option-pricing-service", version="0.1.0",
                  lifespan=lifespan)

    def pricing_service() -> PricingService:
        return app.state.pricing

    # ---------------------------------------------------------------
    # Exception translation
    # ---------------------------------------------------------------
    @app.exception_handler(PricingError)
    async def pricing_error_handler(request, exc: PricingError):
        status = 404 if exc.code is ErrorCode.SESSION_NOT_FOUND else 422
        logger.info("rejected request: code=%s msg=%s detail=%s",
                    exc.code.value, exc.message, exc.detail)
        return JSONResponse(
            status_code=status,
            content={"error_code": exc.code.value,
                     "message": exc.message,
                     "detail": exc.detail},
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request, exc: Exception):
        # Never leak underlying numerical/library exceptions.
        logger.exception("unexpected internal error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error_code": ErrorCode.INTERNAL_ERROR.value,
                     "message": "an unexpected internal error occurred",
                     "detail": {"type": type(exc).__name__}},
        )

    # ---------------------------------------------------------------
    # Meta / built-in definitions
    # ---------------------------------------------------------------
    @app.get("/health", response_model=StatusOut)
    def health():
        return StatusOut(
            status="ok",
            default_as_of=DEFAULT_AS_OF.isoformat(),
            supported_styles=[s.value for s in OptionStyle],
            supported_models=[m.value for m in ModelType],
        )

    @app.get("/v1/catalog")
    def catalog():
        """Built-in market parameters and a small set of example contracts."""
        from app.service import BUILTIN_MARKET
        return {
            "builtin_market": {
                "spot": BUILTIN_MARKET.spot,
                "risk_free_rate": BUILTIN_MARKET.risk_free_rate,
                "dividend_yield": BUILTIN_MARKET.dividend_yield,
            },
            "contracts": [
                {
                    "contract_id": "demo-euro-call-atm",
                    "style": "european", "option_type": "call",
                    "strike": 100.0, "expiry": "2027-09-25",
                },
                {
                    "contract_id": "demo-uoc-120",
                    "style": "barrier", "option_type": "call",
                    "strike": 100.0, "expiry": "2027-09-25",
                    "barrier": 120.0, "barrier_type": "up_and_out",
                    "rebate": 2.0,
                },
            ],
        }

    # ---------------------------------------------------------------
    # Calibration sessions
    # ---------------------------------------------------------------
    @app.post("/v1/sessions", response_model=SessionOut, status_code=201)
    def create_session(payload: CreateSessionRequest):
        calibration: CalibrationService = app.state.calibration
        market = MarketData(
            spot=payload.market.spot,
            risk_free_rate=payload.market.risk_free_rate,
            dividend_yield=payload.market.dividend_yield,
        )
        quotes = [
            MarketQuote(strike=q.strike, expiry=q.expiry,
                        option_type=q.option_type, premium=q.premium)
            for q in payload.quotes
        ]
        session = calibration.create_session(market, quotes, DEFAULT_AS_OF)
        result = session.get_calibration()
        return SessionOut(
            session_id=session.session_id,
            version=session.version,
            calibrated_sigma=result.sigma,
            quote_count=result.quote_count,
        )

    @app.get("/v1/sessions/{session_id}", response_model=SessionOut)
    def get_session(session_id: str):
        session = app.state.calibration.get_session(session_id)
        result = session.get_calibration()
        return SessionOut(
            session_id=session.session_id,
            version=session.version,
            calibrated_sigma=result.sigma,
            quote_count=result.quote_count,
        )

    @app.patch("/v1/sessions/{session_id}/market", response_model=SessionOut)
    def update_market(session_id: str, payload: UpdateMarketRequest):
        session = app.state.calibration.get_session(session_id)
        session.update_market(
            spot=payload.spot,
            risk_free_rate=payload.risk_free_rate,
            dividend_yield=payload.dividend_yield,
        )
        result = session.get_calibration()
        return SessionOut(
            session_id=session.session_id, version=session.version,
            calibrated_sigma=result.sigma, quote_count=result.quote_count,
        )

    @app.put("/v1/sessions/{session_id}/quotes", response_model=SessionOut)
    def update_quotes(session_id: str, payload: UpdateQuotesRequest):
        session = app.state.calibration.get_session(session_id)
        quotes = [
            MarketQuote(strike=q.strike, expiry=q.expiry,
                        option_type=q.option_type, premium=q.premium)
            for q in payload.quotes
        ]
        session.update_quotes(quotes)
        result = session.get_calibration()
        return SessionOut(
            session_id=session.session_id, version=session.version,
            calibrated_sigma=result.sigma, quote_count=result.quote_count,
        )

    @app.delete("/v1/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str):
        app.state.calibration.delete_session(session_id)

    # ---------------------------------------------------------------
    # Pricing (single + batch share the same worker)
    # ---------------------------------------------------------------
    @app.post("/v1/price", response_model=ValuationOut)
    def price_one(payload: PriceRequest):
        if len(payload.items) != 1:
            raise PricingError(
                ErrorCode.EMPTY_BATCH,
                "/v1/price expects exactly one item; use /v1/price/batch "
                "for multiple items",
                {"item_count": len(payload.items)},
            )
        result = pricing_service().price_one(payload.items[0],
                                             payload.session_id)
        return _serialize(result)

    @app.post("/v1/price/batch", response_model=BatchResponse)
    def price_batch(payload: PriceRequest):
        outcomes = pricing_service().price_batch_isolated(
            payload.items, payload.session_id
        )
        entries = []
        for o in outcomes:
            if o["ok"]:
                entries.append(BatchEntry(ok=True, result=_serialize(o["result"])))
            else:
                entries.append(BatchEntry(
                    ok=False, error_code=o["error_code"],
                    message=o["message"], detail=o["detail"],
                ))
        return BatchResponse(count=len(entries), results=entries)

    return app

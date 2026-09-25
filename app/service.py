"""Application orchestration: validation, pricing dispatch and batch jobs.

The single-item and batch entry points share one internal worker, guaranteeing
that a given contract/market/model tuple yields bit-identical results whether
priced alone or inside a batch. Batch jobs run on a bounded thread pool and
calibration sessions deduplicate calibration via single flight.
"""
from __future__ import annotations

import datetime as _dt
import logging
from app.calibration import CalibrationResult, CalibrationService
from app.enums import OptionStyle
from app.errors import ErrorCode, PricingError
from app.models import (
    Greeks,
    MarketData,
    OptionContract,
    PriceItem,
    ValuationResult,
)
from app.pricing.barrier import barrier_price
from app.pricing.black_scholes import bs_greeks, bs_price
from app import validation

logger = logging.getLogger(__name__)

# Centralised default valuation date (fixed for reproducible local service).
DEFAULT_AS_OF = _dt.date(2026, 9, 25)

_FD_BUMP = 1e-4
_FD_BUMP_VOL = 1e-4
_FD_BUMP_RATE = 1e-4
_FD_BUMP_T_DAYS = 1.0


class PricingService:
    def __init__(self, calibration_service: CalibrationService | None = None,
                 max_workers: int = 8) -> None:
        self.calibration_service = calibration_service or CalibrationService(
            max_workers=max_workers
        )
        self._owns_calibration = calibration_service is None
        self._executor = self.calibration_service.executor

    def shutdown(self) -> None:
        if self._owns_calibration:
            self.calibration_service.shutdown()

    # ------------------------------------------------------------------
    # Single item
    # ------------------------------------------------------------------
    def price_one(self, item: PriceItem,
                  session_id: str | None) -> ValuationResult:
        return self._price_item(item, session_id)

    def price_batch(
        self, items: list[PriceItem], session_id: str | None
    ) -> list[ValuationResult | dict]:
        if not items:
            raise PricingError(
                ErrorCode.EMPTY_BATCH, "batch request contains no items"
            )
        # A session is calibrated exactly once for the whole batch: the first
        # task triggers single flight inside the session and all other tasks
        # await/reuse the same immutable result.
        results: list[ValuationResult | dict] = [None] * len(items)  # type: ignore[list-item]

        def work(index: int) -> None:
            results[index] = self._price_item(items[index], session_id)

        # Bounded resource use: the shared pool caps concurrent evaluation.
        futures = [self._executor.submit(work, i) for i in range(len(items))]
        for fut in futures:
            fut.result()  # propagate; per-item errors are captured below
        return results

    def price_batch_isolated(
        self, items: list[PriceItem], session_id: str | None
    ) -> list[dict]:
        """Batch variant that captures per-item errors instead of raising,
        returning serialisable outcome dictionaries aligned by index."""
        if not items:
            raise PricingError(
                ErrorCode.EMPTY_BATCH, "batch request contains no items"
            )
        outcomes: list[dict] = [None] * len(items)  # type: ignore[list-item]

        def work(index: int) -> None:
            try:
                outcomes[index] = {
                    "ok": True,
                    "result": self._price_item(items[index], session_id),
                }
            except PricingError as exc:
                logger.warning(
                    "item %d rejected: code=%s msg=%s",
                    index, exc.code.value, exc.message,
                )
                outcomes[index] = {
                    "ok": False,
                    "error_code": exc.code.value,
                    "message": exc.message,
                    "detail": exc.detail,
                }

        futures = [self._executor.submit(work, i) for i in range(len(items))]
        for fut in futures:
            fut.result()
        return outcomes

    # ------------------------------------------------------------------
    # Core worker (shared by single and batch paths)
    # ------------------------------------------------------------------
    def _price_item(self, item: PriceItem,
                    session_id: str | None) -> ValuationResult:
        contract = self._to_contract(item)
        as_of = item.as_of or DEFAULT_AS_OF

        # 1) structural validation (independent of market)
        validation.validate_contract(contract, as_of)

        # 2) resolve market data + volatility
        calibration: CalibrationResult | None = None
        if session_id is not None:
            session = self.calibration_service.get_session(session_id)
            market = session.market
            if item.sigma_override is not None:
                sigma = float(item.sigma_override)
                calibrated = False
            else:
                calibration = session.get_calibration()
                sigma = calibration.sigma
                calibrated = True
        else:
            # Without a session, an explicit sigma is mandatory because there
            # is nothing to calibrate from.
            if item.sigma_override is None:
                raise PricingError(
                    ErrorCode.INVALID_PARAMETER,
                    "sigma_override is required when session_id is not given",
                )
            market = self._builtin_market()
            sigma = float(item.sigma_override)
            calibrated = False

        validation.validate_market(market)
        validation.validate_volatility(sigma)

        # 3) expiry / model applicability
        tenor = validation.tenor_years(contract.expiry, as_of)
        validation.validate_model(contract, item.model, market.spot)

        logger.info(
            "pricing id=%s style=%s type=%s K=%s H=%s S=%s r=%s q=%s "
            "sigma=%.6f T=%.5f model=%s calibrated=%s",
            contract.contract_id, contract.style.value,
            contract.option_type.value, contract.strike,
            contract.barrier, market.spot, market.risk_free_rate,
            market.dividend_yield, sigma, tenor, item.model.value, calibrated,
        )

        # 4) price + greeks on the single dispatch path
        price, greeks = self._dispatch(contract, market, tenor, sigma)

        return ValuationResult(
            contract_id=contract.contract_id,
            price=price,
            greeks=greeks,
            model=item.model,
            sigma_used=sigma,
            as_of=as_of,
            tenor_years=tenor,
            calibrated=calibrated,
        )

    def _dispatch(
        self,
        contract: OptionContract,
        market: MarketData,
        tenor: float,
        sigma: float,
    ) -> tuple[float, Greeks]:
        S, r, q = market.spot, market.risk_free_rate, market.dividend_yield
        K = contract.strike
        ot = contract.option_type.value

        if contract.style is OptionStyle.EUROPEAN:
            price = bs_price(ot, S, K, tenor, r, q, sigma)
            gk = bs_greeks(ot, S, K, tenor, r, q, sigma)
            return price, Greeks(**gk)

        # Barrier: analytic price; greeks via central finite differences on
        # the same analytic price (consistent by construction).
        assert contract.barrier is not None
        price = barrier_price(
            contract.barrier_type.value, ot, S, K, contract.barrier,
            contract.rebate, tenor, r, q, sigma,
        )
        greeks = self._barrier_greeks(contract, market, tenor, sigma, price)
        return price, greeks

    def _barrier_greeks(
        self,
        contract: OptionContract,
        market: MarketData,
        tenor: float,
        sigma: float,
        price: float,
    ) -> Greeks:
        S, r, q = market.spot, market.risk_free_rate, market.dividend_yield
        H, K = contract.barrier, contract.strike
        ot = contract.barrier_type.value
        opt = contract.option_type.value

        def px(spot=S, rate=r, div=q, vol=sigma, T=tenor):
            return barrier_price(ot, opt, spot, K, H, contract.rebate,
                                 T, rate, div, vol)

        try:
            h = _FD_BUMP * max(S, 1.0)
            delta = (px(spot=S + h) - px(spot=S - h)) / (2 * h)
            gamma = (px(spot=S + h) - 2 * price + px(spot=S - h)) / h ** 2
            vega = (px(vol=sigma + _FD_BUMP_VOL)
                    - px(vol=max(sigma - _FD_BUMP_VOL, 1e-8))) / (2 * _FD_BUMP_VOL)
            rho = (px(rate=r + _FD_BUMP_RATE)
                   - px(rate=r - _FD_BUMP_RATE)) / (2 * _FD_BUMP_RATE)
            # theta = dV/dt = -dV/dT (calendar time)
            t_bump = _FD_BUMP_T_DAYS / validation.DAY_COUNT_DENOMINATOR
            if tenor - t_bump <= 0:
                # One-sided difference near expiry.
                theta = -(px(T=tenor + t_bump) - price) / t_bump
            else:
                theta = -(px(T=tenor + t_bump) - px(T=tenor - t_bump)) / (
                    2 * t_bump
                )
        except PricingError:
            raise
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise PricingError(
                ErrorCode.NUMERICAL_INSTABILITY,
                "finite-difference Greeks became numerically unstable",
                {"contract_id": contract.contract_id,
                 "underlying_error": type(exc).__name__},
            )

        greeks = Greeks(delta=delta, gamma=gamma, vega=vega,
                        theta=theta, rho=rho)
        for name in ("delta", "gamma", "vega", "theta", "rho"):
            value = getattr(greeks, name)
            if not _finite(value):
                raise PricingError(
                    ErrorCode.NUMERICAL_INSTABILITY,
                    f"barrier greek {name} is not finite",
                    {"contract_id": contract.contract_id, name: repr(value)},
                )
        return greeks

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_contract(item: PriceItem) -> OptionContract:
        spec = item.contract
        return OptionContract(
            contract_id=spec.contract_id or "adhoc",
            style=spec.style,
            option_type=spec.option_type,
            strike=float(spec.strike),
            expiry=spec.expiry,
            barrier=None if spec.barrier is None else float(spec.barrier),
            barrier_type=spec.barrier_type,
            rebate=float(spec.rebate),
        )

    @staticmethod
    def _builtin_market() -> MarketData:
        return BUILTIN_MARKET


def _finite(value: float) -> bool:
    return isinstance(value, (int, float)) and value == value and value not in (
        float("inf"), float("-inf"),
    )


# Built-in market parameters used for session-less ad hoc valuations.
BUILTIN_MARKET = MarketData(
    spot=100.0,
    risk_free_rate=0.05,
    dividend_yield=0.02,
)

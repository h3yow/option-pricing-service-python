"""Calibration sessions with concurrency-safe result reuse.

Design
------
* A :class:`CalibrationSession` stores market data, quotes and an immutable
  *calibration result*. Any change to spot/rates/quotes increments a version
  and invalidates the cached result, so stale numbers are never served.
* Calibration fits a single flat Black-Scholes volatility to the quote set
  (equal-weight least squares). It runs at most once per version even under
  concurrent requests (single flight): losers block on the winner and reuse
  the exact same result object.
* The owning :class:`CalibrationService` owns one bounded thread pool shared
  by all sessions and caps the number of retained sessions (resource bound).
"""
from __future__ import annotations

import itertools
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from app.errors import ErrorCode, PricingError
from app.models import MarketData, MarketQuote
from app.pricing.black_scholes import bs_price, implied_vol
from app.validation import (
    validate_market,
    validate_tenor_range,
    validate_volatility,
)

logger = logging.getLogger(__name__)

_SIGMA_FIT_MIN = 1e-4
_SIGMA_FIT_MAX = 5.0
_SSE_REL_TOL = 1e-12
_GOLDEN_GR = (5.0 ** 0.5 - 1.0) / 2.0  # 0.618...
_FIT_MAX_ITER = 200
_FIT_XTOL = 1e-10
# Per-quote implied vols may spread by at most this fraction of their mean;
# beyond this a single flat vol is not a defensible representation.
_MAX_IV_DISPERSION = 0.5


@dataclass(frozen=True)
class CalibrationResult:
    sigma: float
    quote_count: int
    sse: float
    max_abs_residual: float
    version: int
    residuals: tuple[tuple[int, float], ...] = field(default_factory=tuple)


def _model_premium(q: MarketQuote, market: MarketData, tenor: float,
                   sigma: float) -> float:
    return bs_price(
        q.option_type.value,
        market.spot,
        q.strike,
        tenor,
        market.risk_free_rate,
        market.dividend_yield,
        sigma,
    )


def _sse(sigma: float, quotes: list[MarketQuote], tenors: list[float],
         market: MarketData) -> float:
    total = 0.0
    for q, t in zip(quotes, tenors):
        diff = _model_premium(q, market, t, sigma) - q.premium
        total += diff * diff
    return total


def _golden_minimize(f, lo: float, hi: float) -> float:
    """Golden-section minimization of a unimodal scalar function."""
    c = hi - _GOLDEN_GR * (hi - lo)
    d = lo + _GOLDEN_GR * (hi - lo)
    fc, fd = f(c), f(d)
    for _ in range(_FIT_MAX_ITER):
        if abs(hi - lo) < _FIT_XTOL:
            break
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - _GOLDEN_GR * (hi - lo)
            fc = f(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + _GOLDEN_GR * (hi - lo)
            fd = f(d)
    return 0.5 * (lo + hi)


class CalibrationSession:
    """Holds market parameters, quotes and a memoized calibration result."""

    def __init__(self, session_id: str, market: MarketData,
                 quotes: list[MarketQuote], as_of):
        self.session_id = session_id
        self._market = market
        self._quotes = list(quotes)
        self._as_of = as_of
        self._version = 1
        self._result: CalibrationResult | None = None
        self._calib_lock = threading.Lock()
        self._done = threading.Event()
        self._error: BaseException | None = None
        validate_market(market)
        self._validate_quotes()

    # -- introspection -------------------------------------------------
    @property
    def market(self) -> MarketData:
        return self._market

    @property
    def quotes(self) -> tuple[MarketQuote, ...]:
        return tuple(self._quotes)

    @property
    def version(self) -> int:
        return self._version

    # -- mutation ------------------------------------------------------
    def update_market(self, *, spot=None, risk_free_rate=None,
                      dividend_yield=None) -> None:
        new_market = MarketData(
            spot=self._market.spot if spot is None else float(spot),
            risk_free_rate=(self._market.risk_free_rate
                            if risk_free_rate is None else float(risk_free_rate)),
            dividend_yield=(self._market.dividend_yield
                            if dividend_yield is None
                            else float(dividend_yield)),
        )
        validate_market(new_market)
        with self._calib_lock:
            self._market = new_market
            self._invalidate_locked()
        logger.info("session %s market updated -> version %d",
                    self.session_id, self._version)

    def update_quotes(self, quotes: list[MarketQuote]) -> None:
        quotes = list(quotes)
        self._validate_quotes_payload(quotes)
        with self._calib_lock:
            self._quotes = quotes
            self._invalidate_locked()
        logger.info("session %s quotes updated -> version %d",
                    self.session_id, self._version)

    def _invalidate_locked(self) -> None:
        self._version += 1
        self._result = None
        self._error = None
        self._done = threading.Event()

    # -- calibration ---------------------------------------------------
    def get_calibration(self) -> CalibrationResult:
        """Return the calibration for the current version, running it once.

        Concurrent callers share a single calibration (single flight).
        """
        with self._calib_lock:
            if self._result is not None:
                return self._result
            if self._error is not None:
                raise PricingError(
                    ErrorCode.CALIBRATION_DID_NOT_CONVERGE,
                    "calibration previously failed for this version",
                )
            is_leader = not self._done.is_set()
            done = self._done
            if is_leader:
                # Leader performs the work while holding no lock.
                pass
            lock = self._calib_lock

        if is_leader:
            try:
                result = self._run_calibration()
                with lock:
                    self._result = result
                    self._error = None
                    done.set()
                logger.info(
                    "session %s calibrated v%d sigma=%.6f sse=%.3g on %d quotes",
                    self.session_id, result.version, result.sigma,
                    result.sse, result.quote_count,
                )
                return result
            except BaseException as exc:  # propagate identical error to waiters
                with lock:
                    self._error = exc
                    done.set()
                raise

        # Followers wait for the leader, then reuse the immutable result.
        done.wait()
        with lock:
            if self._result is not None:
                return self._result
            err = self._error
        if err is not None:
            if isinstance(err, PricingError):
                raise err
            raise PricingError(
                ErrorCode.INTERNAL_ERROR,
                "calibration failed with an unexpected internal error",
                {"type": type(err).__name__},
            )
        # Defensive: event set but no result/error should never happen.
        raise PricingError(
            ErrorCode.INTERNAL_ERROR,
            "calibration finished without producing a result",
        )

    def _run_calibration(self) -> CalibrationResult:
        from app.validation import tenor_years

        tenors = []
        for q in self._quotes:
            t = tenor_years(q.expiry, self._as_of)
            validate_tenor_range(t)
            tenors.append(t)
        if not self._quotes:
            raise PricingError(ErrorCode.NO_QUOTES, "session has no quotes")

        # Per-quote implied vols: a quote outside the no-arb range has no
        # solution and is reported distinctly.
        ivs = []
        for i, q in enumerate(self._quotes):
            try:
                iv = implied_vol(
                    q.option_type.value, q.premium,
                    self._market.spot, q.strike, tenors[i],
                    self._market.risk_free_rate,
                    self._market.dividend_yield,
                )
            except PricingError:
                raise PricingError(
                    ErrorCode.CALIBRATION_NO_SOLUTION,
                    f"quote #{i} (K={q.strike}, premium={q.premium}) admits "
                    "no implied volatility at current market parameters",
                    {"quote_index": i, "strike": q.strike,
                     "premium": q.premium},
                )
            ivs.append(iv)

        # Consistency check on the per-quote implied vols. A single flat vol
        # is only meaningful when the quotes do not already imply wildly
        # different vols (i.e. no pronounced smile/skew).
        iv_mean = sum(ivs) / len(ivs)
        iv_spread = max(ivs) - min(ivs)
        if iv_mean > 0 and iv_spread / iv_mean > _MAX_IV_DISPERSION:
            raise PricingError(
                ErrorCode.CALIBRATION_DID_NOT_CONVERGE,
                "quotes imply volatilities that differ too much to be "
                "represented by one flat volatility "
                f"(relative spread {iv_spread / iv_mean:.1%} > "
                f"{_MAX_IV_DISPERSION:.0%}); a smile/skew model is required",
                {"iv_min": min(ivs), "iv_max": max(ivs),
                 "relative_spread": iv_spread / iv_mean},
            )

        # Equal-weight least-squares fit of the single flat vol.
        def objective(sigma: float) -> float:
            return _sse(sigma, self._quotes, tenors, self._market)

        sigma = _golden_minimize(
            objective, _SIGMA_FIT_MIN, _SIGMA_FIT_MAX
        )
        validate_volatility(sigma)

        residuals = tuple(
            (i, _model_premium(q, self._market, t, sigma) - q.premium)
            for i, (q, t) in enumerate(zip(self._quotes, tenors))
        )
        sse = sum(d * d for _, d in residuals)
        max_abs = max((abs(d) for _, d in residuals), default=0.0)
        return CalibrationResult(
            sigma=sigma,
            quote_count=len(self._quotes),
            sse=sse,
            max_abs_residual=max_abs,
            version=self._version,
            residuals=residuals,
        )

    # -- payload validation -------------------------------------------
    def _validate_quotes(self) -> None:
        self._validate_quotes_payload(self._quotes)

    @staticmethod
    def _validate_quotes_payload(quotes) -> None:
        if not quotes:
            raise PricingError(ErrorCode.NO_QUOTES,
                               "at least one quote is required to calibrate")
        for i, q in enumerate(quotes):
            if not isinstance(q, MarketQuote):
                raise PricingError(
                    ErrorCode.INVALID_PARAMETER,
                    f"quote #{i} has invalid type",
                )
            if not (isinstance(q.premium, (int, float)) and q.premium > 0):
                raise PricingError(
                    ErrorCode.INVALID_PARAMETER,
                    f"quote #{i} premium must be strictly positive",
                    {"quote_index": i, "premium": repr(q.premium)},
                )
            if not (isinstance(q.strike, (int, float)) and q.strike > 0):
                raise PricingError(
                    ErrorCode.INVALID_PARAMETER,
                    f"quote #{i} strike must be strictly positive",
                    {"quote_index": i, "strike": repr(q.strike)},
                )


class CalibrationService:
    """Manages a bounded collection of sessions and one shared thread pool."""

    def __init__(self, max_sessions: int = 256, max_workers: int = 8):
        self.max_sessions = max_sessions
        self._sessions: dict[str, CalibrationSession] = {}
        self._sessions_lock = threading.Lock()
        self._ids = itertools.count()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="calib"
        )

    @property
    def executor(self) -> ThreadPoolExecutor:
        return self._executor

    def create_session(self, market: MarketData, quotes: list[MarketQuote],
                       as_of) -> CalibrationSession:
        session = CalibrationSession(
            f"sess-{next(self._ids)}", market, quotes, as_of
        )
        with self._sessions_lock:
            if len(self._sessions) >= self.max_sessions:
                raise PricingError(
                    ErrorCode.INVALID_PARAMETER,
                    f"session limit {self.max_sessions} reached; "
                    "close an existing session before creating a new one",
                    {"max_sessions": self.max_sessions},
                )
            self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> CalibrationSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise PricingError(
                ErrorCode.SESSION_NOT_FOUND,
                f"calibration session {session_id!r} does not exist",
                {"session_id": session_id},
            )
        return session

    def delete_session(self, session_id: str) -> None:
        with self._sessions_lock:
            if session_id not in self._sessions:
                raise PricingError(
                    ErrorCode.SESSION_NOT_FOUND,
                    f"calibration session {session_id!r} does not exist",
                    {"session_id": session_id},
                )
            del self._sessions[session_id]

    def active_session_count(self) -> int:
        return len(self._sessions)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)

"""定价服务：校验、模型选择、校准复用与结果组装。

一致性保证：
- 单笔与批量共用同一条 _price_one 计算路径；
- 蒙特卡洛 seed 由 (contract, market, calibrated_vol) 确定性派生，
  与请求入口、并发时序无关，因此同一合约结论逐位一致；
- 校准复用完全交给 CalibrationSession（键含版本与行情快照，更新即重估）；
- 会话定价基于单次原子读取的会话状态：结果要么整体属于更新前内容、
  要么整体属于更新后内容，绝不两半混合。

模型适用范围（双向守住，任何不匹配在出数前拒绝）：
- 欧式：black_scholes；
- 算术平均亚式：monte_carlo；
- 单障碍：barrier_analytic（默认）或 monte_carlo。
"""

from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

from .calibration import CalibrationSession, SessionState
from .contracts import (
    BarrierDirection,
    Contract,
    ExerciseStyle,
    MarketParams,
    Model,
    PricingResult,
)
from .errors import (
    CapacityExceededError,
    ModelNotApplicableError,
    NumericalInstabilityError,
    ValidationError,
)
from .pricing import asian, barrier, black_scholes
from .sessions import SessionManager
from .validation import validate_contract, validate_market

_BATCH_WORKERS = 4      # 批量并发度有界
_BATCH_MAX_ITEMS = 64   # 批量规模上限
_MAX_IN_FLIGHT = 32     # 全局并发估值上限

_DEFAULT_MODEL = {
    ExerciseStyle.EUROPEAN: Model.BLACK_SCHOLES,
    ExerciseStyle.ASIAN_ARITHMETIC: Model.MONTE_CARLO,
    ExerciseStyle.BARRIER_SINGLE: Model.BARRIER_ANALYTIC,
}

_ALLOWED_MODELS = {
    ExerciseStyle.EUROPEAN: frozenset({Model.BLACK_SCHOLES}),
    ExerciseStyle.ASIAN_ARITHMETIC: frozenset({Model.MONTE_CARLO}),
    ExerciseStyle.BARRIER_SINGLE: frozenset(
        {Model.BARRIER_ANALYTIC, Model.MONTE_CARLO}),
}


@dataclass(frozen=True)
class PriceJob:
    """批量估值的一项：合约 + 可选模型 + 可选会话归属。"""

    contract: Contract
    model: Model | None = None
    session_id: str | None = None


def _resolve_model(contract: Contract, model: Model | None) -> Model:
    if model is None:
        return _DEFAULT_MODEL[contract.style]
    if model not in _ALLOWED_MODELS[contract.style]:
        raise ModelNotApplicableError(
            f"模型 {model.value} 不适用于 {contract.style.value} 合约",
            detail={"style": contract.style.value, "model": model.value,
                    "allowed": sorted(m.value
                                      for m in _ALLOWED_MODELS[contract.style])})
    return model


def _derive_seed(contract: Contract, market: MarketParams, vol: float) -> int:
    payload = repr((contract, market, vol)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _check_barrier_not_breached(contract: Contract,
                                market: MarketParams) -> None:
    """已触碰障碍的合约超出解析模型适用范围，出数前明确拒绝。"""
    if contract.style is not ExerciseStyle.BARRIER_SINGLE:
        return
    up = contract.barrier_direction is BarrierDirection.UP
    breached = (market.spot >= contract.barrier if up
                else market.spot <= contract.barrier)
    if breached:
        raise ValidationError(
            "标的价格已越过障碍，连续监测解析模型不再适用",
            detail={"spot": market.spot, "barrier": contract.barrier,
                    "direction": contract.barrier_direction.value})


class PricingService:
    """线程安全的定价入口。单笔与批量共用同一计算路径，保证结论一致。"""

    def __init__(self, session: CalibrationSession,
                 manager: SessionManager | None = None, *,
                 max_batch_items: int = _BATCH_MAX_ITEMS,
                 max_in_flight: int = _MAX_IN_FLIGHT) -> None:
        self._session = session
        self._manager = manager
        self._max_batch_items = max_batch_items
        self._slots = threading.BoundedSemaphore(max_in_flight)

    # ---- 单笔 ---------------------------------------------------------------

    def price(self, contract: Contract, market: MarketParams,
              model: Model | None = None) -> PricingResult:
        return self._price_one(contract, market, model)

    def price_in_session(self, contract: Contract, session_id: str,
                         model: Model | None = None) -> PricingResult:
        """在指定校准会话内估值。结果归属该会话及其状态版本。"""
        session, state = self._session_state(session_id)
        result = self._price_one(contract, state.market, model,
                                 session=session, state=state)
        return replace(result, session_id=session_id,
                       session_version=state.version)

    # ---- 批量 ---------------------------------------------------------------

    def price_batch(self, items: list[tuple[Contract, MarketParams]],
                    model: Model | None = None) -> list[PricingResult]:
        """批量入口：有界线程池并发，结果顺序与输入一致。

        相同 (contract, market) 的校准由会话缓存/single-flight 自动复用。
        """
        self._check_batch_size(len(items))
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=_BATCH_WORKERS) as pool:
            futures = [pool.submit(self._price_one, c, m, model)
                       for c, m in items]
            return [f.result() for f in futures]

    def price_each(self, jobs: list[PriceJob],
                   default_market: MarketParams) -> list[PricingResult]:
        """混合批量入口：每项可归属不同会话，无会话项用 default_market。

        同一会话内相同合约的校准由该会话的缓存/single-flight 复用，
        不会对同一份参数反复校准；跨会话互不共享校准结果。
        """
        self._check_batch_size(len(jobs))
        if not jobs:
            return []

        def run(job: PriceJob) -> PricingResult:
            if job.session_id is None:
                return self._price_one(job.contract, default_market, job.model)
            return self.price_in_session(job.contract, job.session_id,
                                         job.model)

        with ThreadPoolExecutor(max_workers=_BATCH_WORKERS) as pool:
            futures = [pool.submit(run, job) for job in jobs]
            return [f.result() for f in futures]

    # ---- 内部 ---------------------------------------------------------------

    def _check_batch_size(self, n: int) -> None:
        if n > self._max_batch_items:
            raise CapacityExceededError(
                "批量请求规模超过上限",
                detail={"resource": "batch_size",
                        "limit": self._max_batch_items, "got": n})

    def _session_state(self, session_id: str):
        if self._manager is None:
            raise ValidationError("本服务未配置会话管理器",
                                  detail={"session_id": session_id})
        session = self._manager.get(session_id)  # 不存在 -> unknown_session
        return session, session.snapshot()

    def _price_one(self, contract: Contract, market: MarketParams,
                   model: Model | None,
                   session: CalibrationSession | None = None,
                   state: SessionState | None = None) -> PricingResult:
        if not self._slots.acquire(blocking=False):
            raise CapacityExceededError(
                "并发估值请求超过上限，请稍后重试",
                detail={"resource": "concurrency"})
        try:
            return self._price_one_inner(contract, market, model,
                                         session, state)
        finally:
            self._slots.release()

    def _price_one_inner(self, contract: Contract, market: MarketParams,
                         model: Model | None,
                         session: CalibrationSession | None,
                         state: SessionState | None) -> PricingResult:
        validate_contract(contract)
        validate_market(market)
        _check_barrier_not_breached(contract, market)
        resolved = _resolve_model(contract, model)

        if state is not None:
            vol = session.calibrated_volatility_with_state(
                state, contract, market)
        else:
            vol = (session or self._session).calibrated_volatility(
                contract, market)

        if resolved is Model.BLACK_SCHOLES:
            price = black_scholes.bs_price(
                contract.option_type, market.spot, contract.strike,
                contract.maturity_years, market.risk_free_rate, vol)
            greeks = black_scholes.bs_greeks(
                contract.option_type, market.spot, contract.strike,
                contract.maturity_years, market.risk_free_rate, vol)
            diagnostics: dict = {"solver": "closed_form"}
        elif resolved is Model.BARRIER_ANALYTIC:
            price = barrier.barrier_price(
                contract.option_type, contract.barrier_direction,
                contract.knock, market.spot, contract.strike,
                contract.barrier, contract.rebate, contract.maturity_years,
                market.risk_free_rate, vol)
            greeks = barrier.barrier_greeks(
                contract.option_type, contract.barrier_direction,
                contract.knock, market.spot, contract.strike,
                contract.barrier, contract.rebate, contract.maturity_years,
                market.risk_free_rate, vol)
            diagnostics = {"solver": "closed_form"}
        else:  # MONTE_CARLO
            seed = _derive_seed(contract, market, vol)
            if contract.style is ExerciseStyle.ASIAN_ARITHMETIC:
                price, std_err = asian.asian_price(
                    contract.option_type, market.spot, contract.strike,
                    contract.maturity_years, market.risk_free_rate, vol,
                    contract.averaging_points, seed=seed)
                greeks_fn = lambda: asian.asian_greeks(
                    contract.option_type, market.spot, contract.strike,
                    contract.maturity_years, market.risk_free_rate, vol,
                    contract.averaging_points, seed=seed)
            else:
                price, std_err = barrier.barrier_price_mc(
                    contract.option_type, contract.barrier_direction,
                    contract.knock, market.spot, contract.strike,
                    contract.barrier, contract.rebate,
                    contract.maturity_years, market.risk_free_rate, vol,
                    seed=seed)
                greeks_fn = lambda: barrier.barrier_greeks_mc(
                    contract.option_type, contract.barrier_direction,
                    contract.knock, market.spot, contract.strike,
                    contract.barrier, contract.rebate,
                    contract.maturity_years, market.risk_free_rate, vol,
                    seed=seed)
            if std_err > 0.02 * max(price, 1e-12):
                raise NumericalInstabilityError(
                    "蒙特卡洛标准误超过价格 2%，结果不可靠",
                    detail={"price": price, "std_err": std_err})
            greeks = greeks_fn()
            diagnostics = {"solver": "monte_carlo", "seed": seed,
                           "mc_std_err": std_err}

        return PricingResult(
            contract_id=contract.contract_id,
            model=resolved,
            price=price,
            greeks=greeks,
            calibrated_volatility=vol,
            diagnostics=diagnostics,
        )

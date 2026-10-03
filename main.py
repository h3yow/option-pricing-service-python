"""本地期权定价与风险指标服务的 HTTP 入口。

内置行情参数与合约定义；同一合约经 /price 与 /price/batch 的结论一致。
支持按客户/标的创建相互隔离的校准会话（/sessions），会话内估值结果
携带 session_id 与 session_version 以便溯源。
"""

from __future__ import annotations

import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from option_pricing.calibration import CalibrationSession
from option_pricing.contracts import (
    BarrierDirection,
    Contract,
    ExerciseStyle,
    KnockType,
    MarketParams,
    Model,
    OptionType,
    PricingResult,
)
from option_pricing.errors import (
    CalibrationError,
    CapacityExceededError,
    ContractExpiredError,
    ModelNotApplicableError,
    PricingError,
    QuoteMissingError,
    SessionNotFoundError,
    ValidationError,
)
from option_pricing.service import PriceJob, PricingService
from option_pricing.sessions import SessionManager

# ---------------------------------------------------------------- 内置状态

_lock = threading.Lock()
_market = MarketParams(spot=100.0, volatility=0.25, risk_free_rate=0.05)

_CONTRACTS: dict[str, Contract] = {
    "EU-CALL-100": Contract("EU-CALL-100", OptionType.CALL,
                            ExerciseStyle.EUROPEAN, 100.0, 1.0),
    "EU-PUT-100": Contract("EU-PUT-100", OptionType.PUT,
                           ExerciseStyle.EUROPEAN, 100.0, 1.0),
    "ASIAN-CALL-100": Contract("ASIAN-CALL-100", OptionType.CALL,
                               ExerciseStyle.ASIAN_ARITHMETIC, 100.0, 1.0,
                               averaging_points=12),
    "UO-CALL-105": Contract("UO-CALL-105", OptionType.CALL,
                            ExerciseStyle.BARRIER_SINGLE, 100.0, 1.0,
                            barrier=105.0,
                            barrier_direction=BarrierDirection.UP,
                            knock=KnockType.OUT, rebate=3.0),
    "DI-PUT-95": Contract("DI-PUT-95", OptionType.PUT,
                          ExerciseStyle.BARRIER_SINGLE, 100.0, 1.0,
                          barrier=95.0,
                          barrier_direction=BarrierDirection.DOWN,
                          knock=KnockType.IN, rebate=0.0),
    "EXPIRED-CALL": Contract("EXPIRED-CALL", OptionType.CALL,
                             ExerciseStyle.EUROPEAN, 100.0, 0.0),
}

# 内置报价表：各合约在基准波动率 0.25 下的模型价，作为内置校准会话的
# 目标市价。行情更新时会话整体重建，保证新参数下重新校准而非沿用旧结果。
from option_pricing.pricing.barrier import barrier_price as _bp
from option_pricing.pricing.black_scholes import bs_price as _bs

_BASE_VOL = 0.25
_BUILTIN_QUOTES = {
    "EU-CALL-100": _bs(OptionType.CALL, 100.0, 100.0, 1.0, 0.05, _BASE_VOL),
    "EU-PUT-100": _bs(OptionType.PUT, 100.0, 100.0, 1.0, 0.05, _BASE_VOL),
    "ASIAN-CALL-100": _bs(OptionType.CALL, 100.0, 100.0, 1.0, 0.05, _BASE_VOL),
    "UO-CALL-105": _bp(OptionType.CALL, BarrierDirection.UP, KnockType.OUT,
                       100.0, 100.0, 105.0, 3.0, 1.0, 0.05, _BASE_VOL),
    "DI-PUT-95": _bp(OptionType.PUT, BarrierDirection.DOWN, KnockType.IN,
                     100.0, 100.0, 95.0, 0.0, 1.0, 0.05, _BASE_VOL),
}


def _build_service(market: MarketParams) -> PricingService:
    return PricingService(CalibrationSession(dict(_BUILTIN_QUOTES)), _manager)


_manager = SessionManager(_CONTRACTS.get)
_service = _build_service(_market)

app = FastAPI(title="option-pricing-service")


# ---------------------------------------------------------------- 错误映射

_STATUS = {
    ValidationError.code: 422,
    ContractExpiredError.code: 410,
    ModelNotApplicableError.code: 400,
    CalibrationError.code: 422,
    "no_convergence": 422,
    "no_solution": 422,
    "numerical_instability": 422,
    SessionNotFoundError.code: 404,
    QuoteMissingError.code: 422,
    CapacityExceededError.code: 429,
}


@app.exception_handler(PricingError)
async def pricing_error_handler(_request, exc: PricingError):
    status = _STATUS.get(exc.code, 422)
    return _error_response(status, exc)


def _error_response(status: int, exc: PricingError):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status,
        content={"error": {"code": exc.code, "message": exc.message,
                           "detail": exc.detail}},
    )


# ---------------------------------------------------------------- 模式

class PriceRequest(BaseModel):
    contract_id: str
    model: Model | None = None
    session_id: str | None = None  # 指定后在对应校准会话内估值


class BatchItem(BaseModel):
    contract_id: str
    model: Model | None = None
    session_id: str | None = None


class BatchRequest(BaseModel):
    items: list[BatchItem] = Field(min_length=1, max_length=64)


class MarketUpdate(BaseModel):
    spot: float | None = None
    volatility: float | None = None
    risk_free_rate: float | None = None


class MarketIn(BaseModel):
    spot: float
    volatility: float
    risk_free_rate: float


class SessionCreate(BaseModel):
    client: str
    underlying: str
    quotes: dict[str, float]  # contract_id -> 目标市价
    market: MarketIn


class SessionUpdate(BaseModel):
    quotes: dict[str, float] | None = None
    market: MarketIn | None = None


class GreeksOut(BaseModel):
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class PriceOut(BaseModel):
    contract_id: str
    model: Model
    price: float
    greeks: GreeksOut
    calibrated_volatility: float
    diagnostics: dict
    session_id: str | None = None
    session_version: int | None = None


class SessionOut(BaseModel):
    session_id: str
    client: str
    underlying: str
    version: int
    quotes: dict[str, float]
    market: MarketParams


def _to_out(r: PricingResult) -> PriceOut:
    return PriceOut(contract_id=r.contract_id, model=r.model, price=r.price,
                    greeks=GreeksOut(**r.greeks.__dict__),
                    calibrated_volatility=r.calibrated_volatility,
                    diagnostics=r.diagnostics,
                    session_id=r.session_id,
                    session_version=r.session_version)


def _get_contract(contract_id: str) -> Contract:
    try:
        return _CONTRACTS[contract_id]
    except KeyError:
        raise HTTPException(status_code=404,
                            detail={"error": {"code": "unknown_contract",
                                              "message": f"未知合约 {contract_id}",
                                              "detail": {}}})


# ---------------------------------------------------------------- 路由

@app.get("/")
def read_root():
    return {"service": "option-pricing", "contracts": sorted(_CONTRACTS)}


@app.get("/market")
def get_market() -> MarketParams:
    with _lock:
        return _market


@app.post("/market")
def update_market(update: MarketUpdate) -> MarketParams:
    """更新行情参数。任何字段变化都会重建校准会话，强制重新校准。"""
    global _market, _service
    with _lock:
        new = MarketParams(
            spot=update.spot if update.spot is not None else _market.spot,
            volatility=(update.volatility if update.volatility is not None
                        else _market.volatility),
            risk_free_rate=(update.risk_free_rate
                            if update.risk_free_rate is not None
                            else _market.risk_free_rate),
        )
        # 先验证可校准（目标价在新行情下有解），再切换，避免半更新状态
        candidate = _build_service(new)
        candidate.price(_CONTRACTS["EU-CALL-100"], new)
        _market = new
        _service = candidate
        return _market


@app.post("/price", response_model=PriceOut)
def price(req: PriceRequest) -> PriceOut:
    contract = _get_contract(req.contract_id)
    if req.session_id is not None:
        with _lock:
            service = _service
        return _to_out(service.price_in_session(contract, req.session_id,
                                                req.model))
    with _lock:
        service, market = _service, _market
    return _to_out(service.price(contract, market, req.model))


@app.post("/price/batch", response_model=list[PriceOut])
def price_batch(req: BatchRequest) -> list[PriceOut]:
    jobs = [PriceJob(contract=_get_contract(i.contract_id), model=i.model,
                     session_id=i.session_id) for i in req.items]
    with _lock:
        service, market = _service, _market
    results = service.price_each(jobs, market)
    return [_to_out(r) for r in results]


# ---------------------------------------------------------------- 校准会话

@app.post("/sessions", response_model=SessionOut, status_code=201)
def create_session(req: SessionCreate) -> SessionOut:
    for contract_id in req.quotes:
        _get_contract(contract_id)  # 未知合约 -> 404，不创建半成品会话
    market = MarketParams(spot=req.market.spot,
                          volatility=req.market.volatility,
                          risk_free_rate=req.market.risk_free_rate)
    return SessionOut(**_manager.create(req.client, req.underlying,
                                        req.quotes, market).__dict__)


@app.get("/sessions", response_model=list[SessionOut])
def list_sessions() -> list[SessionOut]:
    return [SessionOut(**info.__dict__) for info in _manager.list()]


@app.get("/sessions/{session_id}", response_model=SessionOut)
def get_session(session_id: str) -> SessionOut:
    return SessionOut(**_manager.info(session_id).__dict__)


@app.post("/sessions/{session_id}", response_model=SessionOut)
def update_session(session_id: str, req: SessionUpdate) -> SessionOut:
    """原子更新报价和/或行情；版本递增并强制重新校准。"""
    if req.quotes is None and req.market is None:
        raise HTTPException(status_code=422,
                            detail={"error": {"code": "validation_error",
                                              "message": "更新内容为空",
                                              "detail": {}}})
    if req.quotes is not None:
        for contract_id in req.quotes:
            _get_contract(contract_id)
    market = None
    if req.market is not None:
        market = MarketParams(spot=req.market.spot,
                              volatility=req.market.volatility,
                              risk_free_rate=req.market.risk_free_rate)
    info = _manager.update(session_id, quotes=req.quotes, market=market)
    return SessionOut(**info.__dict__)


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    _manager.delete(session_id)
    return {"deleted": session_id}

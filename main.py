"""本地期权定价与风险指标服务的 HTTP 入口。

内置行情参数与合约定义；同一合约经 /price 与 /price/batch 的结论一致。
"""

from __future__ import annotations

import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from option_pricing.calibration import CalibrationSession
from option_pricing.contracts import (BarrierDirection, Contract,
                                      ExerciseStyle, Greeks, KnockType,
                                      MarketParams, Model, OptionType,
                                      PricingResult)
from option_pricing.errors import (CalibrationError, CapacityExceededError,
                                   ContractExpiredError,
                                   ModelNotApplicableError, PricingError,
                                   QuoteNotFoundError, SessionNotFoundError,
                                   ValidationError)
from option_pricing.service import PricingService
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
    "EXPIRED-CALL": Contract("EXPIRED-CALL", OptionType.CALL,
                             ExerciseStyle.EUROPEAN, 100.0, 0.0),
    # 单障碍合约样例
    "UO-CALL-120": Contract("UO-CALL-120", OptionType.CALL,
                            ExerciseStyle.BARRIER, 100.0, 1.0,
                            barrier=120.0,
                            barrier_direction=BarrierDirection.UP,
                            knock=KnockType.OUT, rebate=2.0),
    "DI-PUT-90": Contract("DI-PUT-90", OptionType.PUT,
                          ExerciseStyle.BARRIER, 100.0, 1.0,
                          barrier=90.0,
                          barrier_direction=BarrierDirection.DOWN,
                          knock=KnockType.IN, rebate=1.0),
}

# 内置目标市价：校准会话据此反解隐含波动率。行情更新时会话整体重建，
# 保证新参数下重新校准而非沿用旧结果。
_TARGET_PRICE = 12.336


def _build_service(market: MarketParams) -> PricingService:
    return PricingService(CalibrationSession(_TARGET_PRICE))


_service = _build_service(_market)
_sessions = SessionManager()

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
    QuoteNotFoundError.code: 422,
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


class BatchItem(BaseModel):
    contract_id: str
    model: Model | None = None


class BatchRequest(BaseModel):
    items: list[BatchItem] = Field(min_length=1, max_length=64)


class MarketUpdate(BaseModel):
    spot: float | None = None
    volatility: float | None = None
    risk_free_rate: float | None = None


class SessionCreate(BaseModel):
    market: MarketParams
    quotes: dict[str, float] = Field(min_length=1)  # contract_id -> 目标价
    session_id: str | None = None


class SessionUpdate(BaseModel):
    market: MarketParams | None = None
    quotes: dict[str, float] | None = Field(default=None, min_length=1)


class SessionOut(BaseModel):
    session_id: str
    version: int
    market: MarketParams
    quotes: dict[str, float]


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
    with _lock:
        service, market = _service, _market
    return _to_out(service.price(contract, market, req.model))


@app.post("/price/batch", response_model=list[PriceOut])
def price_batch(req: BatchRequest) -> list[PriceOut]:
    items = [(_get_contract(i.contract_id), i.model) for i in req.items]
    with _lock:
        service, market = _service, _market
    models = [m for _, m in items]
    contracts = [(c, market) for c, _ in items]
    if len(set(models)) > 1:
        raise HTTPException(status_code=400,
                            detail={"error": {"code": "mixed_models",
                                              "message": "批量请求暂不支持混合模型",
                                              "detail": {}}})
    results = service.price_batch(contracts, models[0])
    return [_to_out(r) for r in results]


# ---------------------------------------------------------------- 会话路由

def _session_out(snap) -> SessionOut:
    return SessionOut(session_id=snap.session_id, version=snap.version,
                      market=snap.market, quotes=dict(snap.quotes))


@app.post("/sessions", response_model=SessionOut, status_code=201)
def create_session(req: SessionCreate) -> SessionOut:
    return _session_out(_sessions.create(req.market, dict(req.quotes),
                                         req.session_id))


@app.get("/sessions/{session_id}", response_model=SessionOut)
def get_session(session_id: str) -> SessionOut:
    return _session_out(_sessions.get(session_id))


@app.put("/sessions/{session_id}", response_model=SessionOut)
def update_session(session_id: str, req: SessionUpdate) -> SessionOut:
    """更新行情/报价：版本 +1 并整体重建校准缓存，后续请求必然重新校准。"""
    return _session_out(_sessions.update(session_id, market=req.market,
                                         quotes=(dict(req.quotes)
                                                 if req.quotes is not None
                                                 else None)))


@app.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str) -> None:
    _sessions.delete(session_id)


@app.post("/sessions/{session_id}/price", response_model=PriceOut)
def price_in_session(session_id: str, req: PriceRequest) -> PriceOut:
    contract = _get_contract(req.contract_id)
    snap = _sessions.get(session_id)  # 原子快照：要么全旧要么全新
    return _to_out(_service.price_in_session(snap, contract, req.model))


@app.post("/sessions/{session_id}/price/batch", response_model=list[PriceOut])
def price_batch_in_session(session_id: str,
                           req: BatchRequest) -> list[PriceOut]:
    items = [(_get_contract(i.contract_id), i.model) for i in req.items]
    models = [m for _, m in items]
    if len(set(models)) > 1:
        raise HTTPException(status_code=400,
                            detail={"error": {"code": "mixed_models",
                                              "message": "批量请求暂不支持混合模型",
                                              "detail": {}}})
    snap = _sessions.get(session_id)
    results = _service.price_batch_in_session(
        snap, [c for c, _ in items], models[0])
    return [_to_out(r) for r in results]

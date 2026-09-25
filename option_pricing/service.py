"""定价服务：校验、模型选择、校准复用与结果组装。

一致性保证：
- 单笔 price() 与批量 price_batch() 共用同一条 _price_one 计算路径；
- 蒙特卡洛 seed 由 (contract, market, calibrated_vol) 确定性派生，
  与请求入口、并发时序无关，因此同一合约结论逐位一致；
- 校准复用完全交给 CalibrationSession（键含行情快照，参数变即重估）。
"""

from __future__ import annotations

import hashlib
import math
from concurrent.futures import ThreadPoolExecutor

from .calibration import CalibrationSession
from .contracts import (Contract, ExerciseStyle, MarketParams, Model,
                        PricingResult)
from .errors import ContractExpiredError, ModelNotApplicableError, ValidationError
from .pricing import asian, black_scholes

_VOL_MIN, _VOL_MAX = 1e-4, 5.0
_RATE_MIN, _RATE_MAX = -0.20, 0.50
_BATCH_WORKERS = 4  # 批量并发度有界


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValidationError(f"参数 {name} 必须为有限数值",
                              detail={"parameter": name, "value": value})


def _validate_contract(contract: Contract) -> None:
    _require_finite("strike", contract.strike)
    if contract.strike <= 0.0:
        raise ValidationError("行权价必须为正",
                              detail={"parameter": "strike",
                                      "value": contract.strike})
    _require_finite("maturity_years", contract.maturity_years)
    if contract.maturity_years <= 0.0:
        raise ContractExpiredError(
            "合约已到期（剩余期限非正），无法估值",
            detail={"maturity_years": contract.maturity_years})
    if contract.style is ExerciseStyle.ASIAN_ARITHMETIC:
        if not (1 <= contract.averaging_points <= 366):
            raise ValidationError(
                "亚式期权平均点数量须在 [1, 366]",
                detail={"parameter": "averaging_points",
                        "value": contract.averaging_points})


def _validate_market(market: MarketParams) -> None:
    _require_finite("spot", market.spot)
    if market.spot <= 0.0:
        raise ValidationError("标的价格必须为正",
                              detail={"parameter": "spot", "value": market.spot})
    _require_finite("volatility", market.volatility)
    if not (_VOL_MIN <= market.volatility <= _VOL_MAX):
        raise ValidationError(
            f"波动率须在 [{_VOL_MIN}, {_VOL_MAX}] 内",
            detail={"parameter": "volatility", "value": market.volatility})
    _require_finite("risk_free_rate", market.risk_free_rate)
    if not (_RATE_MIN <= market.risk_free_rate <= _RATE_MAX):
        raise ValidationError(
            f"无风险利率须在 [{_RATE_MIN}, {_RATE_MAX}] 内",
            detail={"parameter": "risk_free_rate",
                    "value": market.risk_free_rate})


def _resolve_model(contract: Contract, model: Model | None) -> Model:
    if model is None:
        return (Model.BLACK_SCHOLES if contract.style is ExerciseStyle.EUROPEAN
                else Model.MONTE_CARLO)
    if (contract.style is ExerciseStyle.ASIAN_ARITHMETIC
            and model is Model.BLACK_SCHOLES):
        raise ModelNotApplicableError(
            "Black-Scholes 解析式不适用于路径相关的亚式期权",
            detail={"style": contract.style.value, "model": model.value})
    return model


def _derive_seed(contract: Contract, market: MarketParams, vol: float) -> int:
    payload = repr((contract, market, vol)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


class PricingService:
    """线程安全的定价入口。单笔与批量共用同一计算路径，保证结论一致。"""

    def __init__(self, session: CalibrationSession) -> None:
        self._session = session

    def price(self, contract: Contract, market: MarketParams,
              model: Model | None = None) -> PricingResult:
        return self._price_one(contract, market, model)

    def price_batch(self, items: list[tuple[Contract, MarketParams]],
                    model: Model | None = None) -> list[PricingResult]:
        """批量入口：有界线程池并发，结果顺序与输入一致。

        相同 (contract, market) 的校准由会话缓存/single-flight 自动复用。
        """
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=_BATCH_WORKERS) as pool:
            futures = [pool.submit(self._price_one, c, m, model)
                       for c, m in items]
            return [f.result() for f in futures]

    def _price_one(self, contract: Contract, market: MarketParams,
                   model: Model | None) -> PricingResult:
        _validate_contract(contract)
        _validate_market(market)
        resolved = _resolve_model(contract, model)

        vol = self._session.calibrated_volatility(contract, market)

        if resolved is Model.BLACK_SCHOLES:
            price = black_scholes.bs_price(
                contract.option_type, market.spot, contract.strike,
                contract.maturity_years, market.risk_free_rate, vol)
            greeks = black_scholes.bs_greeks(
                contract.option_type, market.spot, contract.strike,
                contract.maturity_years, market.risk_free_rate, vol)
            diagnostics: dict = {"solver": "closed_form"}
        else:
            seed = _derive_seed(contract, market, vol)
            price, std_err = asian.asian_price(
                contract.option_type, market.spot, contract.strike,
                contract.maturity_years, market.risk_free_rate, vol,
                contract.averaging_points, seed=seed)
            if std_err > 0.02 * max(price, 1e-12):
                from .errors import NumericalInstabilityError
                raise NumericalInstabilityError(
                    "蒙特卡洛标准误超过价格 2%，结果不可靠",
                    detail={"price": price, "std_err": std_err})
            greeks = asian.asian_greeks(
                contract.option_type, market.spot, contract.strike,
                contract.maturity_years, market.risk_free_rate, vol,
                contract.averaging_points, seed=seed)
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

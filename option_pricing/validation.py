"""合约与行情参数的集中校验，供服务层、校准会话与接口层共用。

所有非法输入在这里被归类为可区分错误，绝不向定价引擎传递越界参数。
"""

from __future__ import annotations

import math

from .contracts import Contract, ExerciseStyle, MarketParams
from .errors import ContractExpiredError, ValidationError

VOL_MIN, VOL_MAX = 1e-4, 5.0
RATE_MIN, RATE_MAX = -0.20, 0.50


def require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValidationError(f"参数 {name} 必须为有限数值",
                              detail={"parameter": name, "value": value})


def validate_contract(contract: Contract) -> None:
    require_finite("strike", contract.strike)
    if contract.strike <= 0.0:
        raise ValidationError("行权价必须为正",
                              detail={"parameter": "strike",
                                      "value": contract.strike})
    require_finite("maturity_years", contract.maturity_years)
    if contract.maturity_years <= 0.0:
        raise ContractExpiredError(
            "合约已到期（剩余期限非正），无法估值",
            detail={"maturity_years": contract.maturity_years})
    if contract.style is ExerciseStyle.ASIAN_ARITHMETIC and not (
            1 <= contract.averaging_points <= 366):
        raise ValidationError(
            "亚式期权平均点数量须在 [1, 366]",
            detail={"parameter": "averaging_points",
                    "value": contract.averaging_points})
    if contract.style is ExerciseStyle.BARRIER_SINGLE:
        if contract.barrier is None or contract.barrier_direction is None \
                or contract.knock is None:
            raise ValidationError(
                "障碍合约必须给出 barrier / barrier_direction / knock",
                detail={"parameter": "barrier"})
        require_finite("barrier", contract.barrier)
        if contract.barrier <= 0.0:
            raise ValidationError("障碍水平必须为正",
                                  detail={"parameter": "barrier",
                                          "value": contract.barrier})
        require_finite("rebate", contract.rebate)
        if contract.rebate < 0.0:
            raise ValidationError("回扣不得为负",
                                  detail={"parameter": "rebate",
                                          "value": contract.rebate})


def validate_market(market: MarketParams) -> None:
    require_finite("spot", market.spot)
    if market.spot <= 0.0:
        raise ValidationError("标的价格必须为正",
                              detail={"parameter": "spot", "value": market.spot})
    require_finite("volatility", market.volatility)
    if not (VOL_MIN <= market.volatility <= VOL_MAX):
        raise ValidationError(
            f"波动率须在 [{VOL_MIN}, {VOL_MAX}] 内",
            detail={"parameter": "volatility", "value": market.volatility})
    require_finite("risk_free_rate", market.risk_free_rate)
    if not (RATE_MIN <= market.risk_free_rate <= RATE_MAX):
        raise ValidationError(
            f"无风险利率须在 [{RATE_MIN}, {RATE_MAX}] 内",
            detail={"parameter": "risk_free_rate",
                    "value": market.risk_free_rate})

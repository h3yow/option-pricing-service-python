"""合约定义、行情参数与结果类型。"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class OptionType(str, enum.Enum):
    CALL = "call"
    PUT = "put"


class ExerciseStyle(str, enum.Enum):
    EUROPEAN = "european"
    ASIAN_ARITHMETIC = "asian_arithmetic"  # 路径相关：算术平均亚式期权


class Model(str, enum.Enum):
    BLACK_SCHOLES = "black_scholes"
    MONTE_CARLO = "monte_carlo"


@dataclass(frozen=True)
class Contract:
    """期权合约定义。frozen 保证可哈希、可安全并发共享。"""

    contract_id: str
    option_type: OptionType
    style: ExerciseStyle
    strike: float
    maturity_years: float  # 以年为单位的剩余期限
    averaging_points: int = 12  # 仅亚式期权使用


@dataclass(frozen=True)
class MarketParams:
    """内置行情参数快照。任何字段变化都会得到新的快照对象。"""

    spot: float
    volatility: float
    risk_free_rate: float
    as_of: str = "2026-09-25"


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


@dataclass(frozen=True)
class PricingResult:
    contract_id: str
    model: Model
    price: float
    greeks: Greeks
    calibrated_volatility: float  # 本次估值实际使用的（校准后）波动率
    diagnostics: dict = field(default_factory=dict)

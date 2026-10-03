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
    BARRIER = "barrier"  # 单障碍欧式期权


class BarrierDirection(str, enum.Enum):
    UP = "up"
    DOWN = "down"


class KnockType(str, enum.Enum):
    IN = "in"
    OUT = "out"


class Model(str, enum.Enum):
    BLACK_SCHOLES = "black_scholes"
    MONTE_CARLO = "monte_carlo"
    BARRIER_ANALYTIC = "barrier_analytic"  # Reiner-Rubinstein 解析式


@dataclass(frozen=True)
class Contract:
    """期权合约定义。frozen 保证可哈希、可安全并发共享。"""

    contract_id: str
    option_type: OptionType
    style: ExerciseStyle
    strike: float
    maturity_years: float  # 以年为单位的剩余期限
    averaging_points: int = 12  # 仅亚式期权使用
    # 以下字段仅单障碍合约（style=BARRIER）使用
    barrier: float | None = None
    barrier_direction: BarrierDirection | None = None
    knock: KnockType | None = None
    rebate: float = 0.0  # 敲出/敲入触发时的回扣（到期支付，折现计入价格）


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
    # 会话来源：非会话路径为 None；会话路径标识出自哪个会话的哪一版内容
    session_id: str | None = None
    session_version: int | None = None

"""按客户/按标的的独立校准会话管理。

一致性设计：
- 每个会话的内容（报价集合 + 行情快照 + 校准缓存）封装在一个不可变的
  SessionSnapshot 中；更新 = 构造全新快照并在锁内原子替换引用。
  并发读取拿到的永远是完整旧版或完整新版，绝不出现两半拼接。
- 更新后版本号单调递增，且校准缓存整体重建：后续请求必然基于新内容
  重新校准，不可能沿用更新前的结果。
- 会话之间完全隔离：各自持有独立的 CalibrationSession 与缓存，一个
  会话的校准结果不会被另一个会话复用。
- 容量有界：会话总数、每会话校准缓存条目均有上限，到顶按可区分原因
  拒绝（CapacityExceededError），不允许无限增长。
"""

from __future__ import annotations

import math
import threading
import uuid
from dataclasses import dataclass, field
from types import MappingProxyType

from .calibration import CalibrationSession
from .contracts import Contract, MarketParams
from .errors import (CapacityExceededError, QuoteNotFoundError,
                     SessionNotFoundError, ValidationError)

_MAX_SESSIONS = 32          # 会话总数上限
_MAX_CACHE_ENTRIES = 128    # 每会话校准缓存条目上限
_MAX_QUOTES = 64            # 每会话报价条数上限


class _QuoteCalibration(CalibrationSession):
    """按合约从本会话报价集合取目标价的校准会话。"""

    def __init__(self, quotes: dict[str, float], *,
                 max_entries: int) -> None:
        super().__init__(None, max_entries=max_entries)
        self._quotes = quotes

    def _target_for(self, contract: Contract) -> float:
        try:
            return self._quotes[contract.contract_id]
        except KeyError:
            raise QuoteNotFoundError(
                "本会话没有该合约的报价，无法校准",
                detail={"contract_id": contract.contract_id})


@dataclass(frozen=True)
class SessionSnapshot:
    """会话内容的不可变整体。读取方拿到的一定是完整的一致版本。"""

    session_id: str
    version: int  # 创建为 1，每次更新 +1
    market: MarketParams
    quotes: MappingProxyType  # contract_id -> 目标价（只读视图）
    calibration: _QuoteCalibration = field(compare=False)


def _validate_market_snapshot(market: MarketParams) -> None:
    for name, value in (("spot", market.spot),
                        ("volatility", market.volatility),
                        ("risk_free_rate", market.risk_free_rate)):
        if not math.isfinite(value):
            raise ValidationError(f"行情参数 {name} 必须为有限数值",
                                  detail={"parameter": name, "value": value})
    if market.spot <= 0.0:
        raise ValidationError("标的价格必须为正",
                              detail={"parameter": "spot", "value": market.spot})
    if not (1e-4 <= market.volatility <= 5.0):
        raise ValidationError("波动率须在 [1e-4, 5.0] 内",
                              detail={"parameter": "volatility",
                                      "value": market.volatility})
    if not (-0.20 <= market.risk_free_rate <= 0.50):
        raise ValidationError("无风险利率须在 [-0.20, 0.50] 内",
                              detail={"parameter": "risk_free_rate",
                                      "value": market.risk_free_rate})


def _validate_quotes(quotes: dict[str, float]) -> None:
    if not quotes:
        raise ValidationError("会话报价集合不能为空", detail={})
    if len(quotes) > _MAX_QUOTES:
        raise CapacityExceededError(
            "会话报价条数超过上限",
            detail={"resource": "quotes_per_session",
                    "limit": _MAX_QUOTES, "actual": len(quotes)})
    for contract_id, price in quotes.items():
        if not math.isfinite(price) or price <= 0.0:
            raise ValidationError(
                "报价必须为正有限值",
                detail={"contract_id": contract_id, "quote": price})


class SessionManager:
    """会话生命周期管理。线程安全；容量有界。"""

    def __init__(self, *, max_sessions: int = _MAX_SESSIONS,
                 max_cache_entries: int = _MAX_CACHE_ENTRIES) -> None:
        self._max_sessions = max_sessions
        self._max_cache_entries = max_cache_entries
        self._lock = threading.Lock()
        self._states: dict[str, SessionSnapshot] = {}

    @property
    def max_sessions(self) -> int:
        return self._max_sessions

    def count(self) -> int:
        with self._lock:
            return len(self._states)

    def create(self, market: MarketParams, quotes: dict[str, float],
               session_id: str | None = None) -> SessionSnapshot:
        """创建会话。校验失败或容量到顶时不产生任何状态。"""
        _validate_market_snapshot(market)
        _validate_quotes(quotes)
        sid = session_id or f"sess-{uuid.uuid4().hex[:12]}"
        with self._lock:
            if sid in self._states:
                raise ValidationError(
                    "会话 ID 已存在", detail={"session_id": sid})
            if len(self._states) >= self._max_sessions:
                raise CapacityExceededError(
                    "会话数量达到上限，拒绝创建",
                    detail={"resource": "sessions",
                            "limit": self._max_sessions})
            snap = self._build(sid, 1, market, quotes)
            self._states[sid] = snap
            return snap

    def get(self, session_id: str) -> SessionSnapshot:
        """取当前快照（原子引用读取，要么全旧要么全新）。"""
        with self._lock:
            return self._get_locked(session_id)

    def _get_locked(self, session_id: str) -> SessionSnapshot:
        try:
            return self._states[session_id]
        except KeyError:
            raise SessionNotFoundError(
                "会话不存在或已删除",
                detail={"session_id": session_id})

    def update(self, session_id: str, *,
               market: MarketParams | None = None,
               quotes: dict[str, float] | None = None) -> SessionSnapshot:
        """更新行情和/或报价：版本 +1，校准缓存整体重建（强制重新校准）。

        先在锁外构造并校验新快照，再在锁内原子替换；替换失败（会话已被
        删除）时不留半更新状态。
        """
        with self._lock:
            old = self._get_locked(session_id)
        new_market = market if market is not None else old.market
        new_quotes = (dict(quotes) if quotes is not None
                      else dict(old.quotes))
        _validate_market_snapshot(new_market)
        _validate_quotes(new_quotes)
        snap = self._build(session_id, old.version + 1, new_market, new_quotes)
        with self._lock:
            if session_id not in self._states:
                raise SessionNotFoundError(
                    "会话在更新期间被删除",
                    detail={"session_id": session_id})
            self._states[session_id] = snap
        return snap

    def delete(self, session_id: str) -> None:
        with self._lock:
            if self._states.pop(session_id, None) is None:
                raise SessionNotFoundError(
                    "会话不存在或已删除",
                    detail={"session_id": session_id})

    def _build(self, session_id: str, version: int, market: MarketParams,
               quotes: dict[str, float]) -> SessionSnapshot:
        return SessionSnapshot(
            session_id=session_id,
            version=version,
            market=market,
            quotes=MappingProxyType(dict(quotes)),
            calibration=_QuoteCalibration(
                dict(quotes), max_entries=self._max_cache_entries),
        )

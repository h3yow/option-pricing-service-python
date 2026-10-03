"""校准会话注册表：按客户/标的维护相互隔离的校准会话。

- 每个会话持有自己的报价表与行情快照（见 CalibrationSession），会话之间
  不共享任何缓存或校准结果。
- 会话数量有硬上限，到顶以 capacity_exceeded(resource=sessions) 拒绝。
- 创建 / 查询 / 更新 / 删除均为线程安全操作；更新语义（原子、版本递增、
  强制重校准）由 CalibrationSession 保证。
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .calibration import CalibrationSession
from .contracts import Contract, MarketParams
from .errors import CapacityExceededError, SessionNotFoundError, ValidationError
from .validation import validate_market

MAX_SESSIONS = 64            # 会话数量硬上限
SESSION_CACHE_ENTRIES = 128  # 每个会话的校准缓存上限


@dataclass(frozen=True)
class SessionInfo:
    """会话的对外只读视图。"""

    session_id: str
    client: str
    underlying: str
    version: int
    quotes: dict[str, float]
    market: MarketParams


class SessionManager:
    """线程安全的会话注册表。容量有界，会话间完全隔离。"""

    def __init__(self, contract_resolver: Callable[[str], Contract | None],
                 *, max_sessions: int = MAX_SESSIONS,
                 max_entries: int = SESSION_CACHE_ENTRIES) -> None:
        self._resolver = contract_resolver
        self._max_sessions = max_sessions
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._sessions: dict[str, tuple[str, str, CalibrationSession]] = {}
        self._counter = itertools.count(1)

    # ---- 生命周期 ----------------------------------------------------------

    def create(self, client: str, underlying: str,
               quotes: Mapping[str, float],
               market: MarketParams) -> SessionInfo:
        if not client or not underlying:
            raise ValidationError("客户与标的标识不能为空",
                                  detail={"client": client,
                                          "underlying": underlying})
        validate_market(market)
        session = CalibrationSession(
            quotes, market=market, max_entries=self._max_entries,
            contract_resolver=self._resolver)
        # 创建即试算：报价在初始行情下不可解时拒绝创建，不留半成品会话
        state = session.snapshot()
        if isinstance(state.quotes, Mapping):
            for contract_id in state.quotes:
                session._trial_solve(state, contract_id)
        with self._lock:
            if len(self._sessions) >= self._max_sessions:
                raise CapacityExceededError(
                    "校准会话数量已到上限",
                    detail={"resource": "sessions",
                            "limit": self._max_sessions})
            session_id = f"S{next(self._counter):04d}"
            self._sessions[session_id] = (client, underlying, session)
        return self._info(session_id, client, underlying, session)

    def get(self, session_id: str) -> CalibrationSession:
        with self._lock:
            try:
                return self._sessions[session_id][2]
            except KeyError:
                raise SessionNotFoundError(
                    "校准会话不存在或已删除",
                    detail={"session_id": session_id}) from None

    def info(self, session_id: str) -> SessionInfo:
        with self._lock:
            try:
                client, underlying, session = self._sessions[session_id]
            except KeyError:
                raise SessionNotFoundError(
                    "校准会话不存在或已删除",
                    detail={"session_id": session_id}) from None
        return self._info(session_id, client, underlying, session)

    def list(self) -> list[SessionInfo]:
        with self._lock:
            items = [(sid, c, u, s) for sid, (c, u, s) in self._sessions.items()]
        return [self._info(*item) for item in items]

    def update(self, session_id: str, *,
               quotes: Mapping[str, float] | None = None,
               market: MarketParams | None = None) -> SessionInfo:
        """原子更新会话内容并强制重校准（见 CalibrationSession.update）。"""
        with self._lock:
            try:
                client, underlying, session = self._sessions[session_id]
            except KeyError:
                raise SessionNotFoundError(
                    "校准会话不存在或已删除",
                    detail={"session_id": session_id}) from None
        session.update(quotes=quotes, market=market)
        return self._info(session_id, client, underlying, session)

    def delete(self, session_id: str) -> None:
        with self._lock:
            try:
                del self._sessions[session_id]
            except KeyError:
                raise SessionNotFoundError(
                    "校准会话不存在或已删除",
                    detail={"session_id": session_id}) from None

    # ---- 内部 ---------------------------------------------------------------

    def _info(self, session_id: str, client: str, underlying: str,
              session: CalibrationSession) -> SessionInfo:
        state = session.snapshot()
        quotes = (dict(state.quotes) if isinstance(state.quotes, Mapping)
                  else {"*": state.quotes})
        return SessionInfo(session_id=session_id, client=client,
                           underlying=underlying, version=state.version,
                           quotes=quotes, market=state.market)

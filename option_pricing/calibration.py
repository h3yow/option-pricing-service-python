"""校准会话：由目标市价反解隐含波动率，并按行情参数快照缓存复用。

约定：
- 隐含波动率定义：欧式/亚式合约按市场惯例取 Black-Scholes 隐含波动率；
  障碍合约取使其障碍解析式价格等于目标报价的波动率（对该合约本身成立）。
- 报价（quotes）可以是单一目标价（向后兼容，适用于任意合约），也可以是
  按 contract_id 索引的报价表；报价表模式下校准只认本会话的报价，
  缺失报价以 quote_missing 明确拒绝。
- 会话可绑定一份行情快照；缓存键为 (版本, 合约, 行情) 整体：报价或行情
  更新都会递增版本并清空缓存，强制重新校准，绝不沿用旧结果。
- 更新是原子的：先在候选状态上试算全部报价，全部可解才一次性切换；
  并发读取要么拿到更新前完整状态、要么拿到更新后完整状态，不会两半混合。
- 并发同一键采用 single-flight：只有一个线程真正求解，其余等待共享结果。
- 缓存为容量有界的 LRU，资源占用可预期。
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from collections.abc import Mapping
from types import MappingProxyType
from typing import NamedTuple

from .contracts import Contract, ExerciseStyle, MarketParams
from .errors import (
    NoConvergenceError,
    NoSolutionError,
    NumericalInstabilityError,
    QuoteMissingError,
    ValidationError,
)
from .pricing.barrier import barrier_price
from .pricing.black_scholes import bs_price
from .validation import validate_market

_VOL_LO, _VOL_HI = 1e-4, 5.0
_GRID_POINTS = 64  # 符号扫描网格点数（对数等距）


def _model_price(contract: Contract, market: MarketParams, vol: float) -> float:
    """校准目标函数：欧式/亚式用 BS 解析式（市场惯例的 BS 隐含波动率）；
    障碍合约用障碍解析式，使隐含波动率对该合约本身成立。"""
    if contract.style is ExerciseStyle.BARRIER_SINGLE:
        return barrier_price(contract.option_type, contract.barrier_direction,
                             contract.knock, market.spot, contract.strike,
                             contract.barrier, contract.rebate,
                             contract.maturity_years, market.risk_free_rate,
                             vol)
    return bs_price(contract.option_type, market.spot, contract.strike,
                    contract.maturity_years, market.risk_free_rate, vol)


def _vol_lo_for(contract: Contract, market: MarketParams) -> float:
    """校准波动率下限。障碍解析式含 (H/S)^{2μ}、μ≈(r-σ²/2)/σ²，极低
    波动率下幂次溢出；按 |2μ·ln(H/S)| ≤ 600 自适应抬升下限。"""
    if contract.style is not ExerciseStyle.BARRIER_SINGLE:
        return _VOL_LO
    log_hs = abs(math.log(contract.barrier / market.spot))
    if log_hs == 0.0:
        return _VOL_LO
    bound = math.sqrt(2.0 * (abs(market.risk_free_rate) + 0.5)
                      * log_hs / 600.0)
    return max(_VOL_LO, bound)


def _brentq(f, lo: float, hi: float, *, tol: float, max_iter: int) -> float:
    """Brent 求根。f(lo) 与 f(hi) 必须异号（由调用方保证）。"""
    flo, fhi = f(lo), f(hi)
    for value, name in ((flo, "f(vol_lo)"), (fhi, "f(vol_hi)")):
        if not math.isfinite(value):
            raise NumericalInstabilityError(
                "校准目标函数在区间端点非有限值", detail={"quantity": name})
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi

    a, b, fa, fb = lo, hi, flo, fhi
    c, fc = a, fa
    d = e = b - a
    for _ in range(max_iter):
        if fb * fc > 0.0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol1 = 2.0 * math.ulp(1.0) * abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)
        if abs(xm) <= tol1 or fb == 0.0:
            return b
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * xm * q - abs(tol1 * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = xm
        else:
            d = e = xm
        a, fa = b, fb
        b += d if abs(d) > tol1 else (tol1 if xm > 0 else -tol1)
        fb = f(b)
        if not math.isfinite(fb):
            raise NumericalInstabilityError(
                "校准目标函数在迭代中出现非有限值",
                detail={"quantity": "f(vol)", "vol": b})
    raise NoConvergenceError(
        f"隐含波动率求解在 {max_iter} 次迭代内未收敛",
        detail={"max_iter": max_iter, "tol": tol})


def _normalize_quotes(quotes: float | Mapping[str, float]):
    """校验并固化报价。返回 float（单一目标价）或只读映射。"""
    if isinstance(quotes, Mapping):
        if not quotes:
            raise ValidationError("报价表不能为空", detail={"quotes": "empty"})
        frozen = {}
        for contract_id, target in quotes.items():
            _validate_target(target, contract_id=contract_id)
            frozen[str(contract_id)] = float(target)
        return MappingProxyType(frozen)
    _validate_target(quotes)
    return float(quotes)


def _validate_target(target: float, *, contract_id: str | None = None) -> None:
    if not isinstance(target, (int, float)) or not math.isfinite(target) \
            or target <= 0.0:
        raise ValidationError(
            "校准目标价格必须为正有限值",
            detail={"target_price": target, "contract_id": contract_id})


class SessionState(NamedTuple):
    """会话内容的不可变快照。读取方拿到后即与后续更新完全隔离。"""

    quotes: object          # float（单一目标价）或 MappingProxyType
    market: MarketParams | None
    version: int


class _InFlight:
    __slots__ = ("error", "event", "result")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: float | None = None
        self.error: Exception | None = None


class CalibrationSession:
    """一次校准会话。线程安全；缓存容量有界（LRU）；更新原子生效。"""

    def __init__(self, quotes: float | Mapping[str, float], *,
                 market: MarketParams | None = None, max_entries: int = 128,
                 max_iter: int = 100, tol: float = 1e-10,
                 contract_resolver=None) -> None:
        self._max_entries = max_entries
        self._max_iter = max_iter
        self._tol = tol
        # contract_resolver: contract_id -> Contract | None；报价表模式下
        # 用于更新前试算（未知合约或不可达报价 -> 拒绝更新）
        self._contract_resolver = contract_resolver
        self._lock = threading.Lock()
        self._update_lock = threading.Lock()  # 串行化更新，保证版本号线性递增
        self._state = SessionState(_normalize_quotes(quotes), market, 0)
        self._cache: OrderedDict[tuple[int, Contract, MarketParams], float] \
            = OrderedDict()
        self._inflight: dict[tuple[int, Contract, MarketParams], _InFlight] \
            = {}

    # ---- 状态访问 ----------------------------------------------------------

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def version(self) -> int:
        return self._state.version

    def snapshot(self) -> SessionState:
        """返回当前不可变状态；单次原子读取，绝不返回两半混合的内容。"""
        return self._state

    # ---- 更新 --------------------------------------------------------------

    def update(self, *, quotes: float | Mapping[str, float] | None = None,
               market: MarketParams | None = None) -> int:
        """原子更新报价和/或行情，返回新版本号。

        先构造候选状态并试算全部报价（在候选行情下全部可解），再一次性
        切换并清空缓存；任何一步失败都保持旧状态不变。更新串行化，
        版本号线性递增。
        """
        with self._update_lock:
            old = self._state
            new_quotes = old.quotes if quotes is None \
                else _normalize_quotes(quotes)
            new_market = old.market if market is None else market
            if new_market is not None:
                validate_market(new_market)
            candidate = SessionState(new_quotes, new_market, old.version + 1)
            # 试算：报价表模式下每个合约都要在候选行情下有解
            if isinstance(new_quotes, Mapping) and new_market is not None:
                for contract_id in new_quotes:
                    self._trial_solve(candidate, contract_id)
            with self._lock:
                self._state = candidate
                self._cache.clear()
            return candidate.version

    def _trial_solve(self, state: SessionState, contract_id: str) -> None:
        """用注入的合约查询器试算；未注入时跳过（可解性顺延到校准时判定）。"""
        if self._contract_resolver is None:
            return
        contract = self._contract_resolver(contract_id)
        if contract is None:
            raise ValidationError(
                "报价引用了未知合约，更新被拒绝",
                detail={"contract_id": contract_id})
        self._solve(state, contract, state.market)

    # ---- 校准 --------------------------------------------------------------

    def calibrated_volatility(self, contract: Contract,
                              market: MarketParams | None = None) -> float:
        """返回该合约的隐含波动率；失败时抛出可区分错误。"""
        return self._calibrated(self._state, contract, market)

    def calibrated_volatility_versioned(
            self, contract: Contract,
            market: MarketParams | None = None) -> tuple[float, int]:
        """与 calibrated_volatility 相同，但同时返回所用状态的版本号。

        状态在单次原子读取中确定，因此返回的 (波动率, 版本) 必然同源。
        """
        state = self._state
        return self._calibrated(state, contract, market), state.version

    def calibrated_volatility_with_state(
            self, state: SessionState, contract: Contract,
            market: MarketParams | None = None) -> float:
        """在调用方原子读取的状态快照上校准，保证与市场/版本同源。"""
        return self._calibrated(state, contract, market)

    def _calibrated(self, state: SessionState, contract: Contract,
                    market: MarketParams | None) -> float:
        market = market if market is not None else state.market
        if market is None:
            raise ValidationError(
                "会话未绑定行情快照，且调用方未提供行情",
                detail={"parameter": "market"})
        key = (state.version, contract, market)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            flight = self._inflight.get(key)
            if flight is None:
                flight = _InFlight()
                self._inflight[key] = flight
                owner = True
            else:
                owner = False

        if not owner:
            flight.event.wait()
            if flight.error is not None:
                raise flight.error
            assert flight.result is not None
            return flight.result

        try:
            vol = self._solve(state, contract, market)
        except Exception as exc:
            flight.error = exc
            raise
        else:
            flight.result = vol
            with self._lock:
                self._cache[key] = vol
                self._cache.move_to_end(key)
                while len(self._cache) > self._max_entries:
                    self._cache.popitem(last=False)
            return vol
        finally:
            with self._lock:
                self._inflight.pop(key, None)
            flight.event.set()

    def _target_for(self, state: SessionState, contract: Contract) -> float:
        if isinstance(state.quotes, Mapping):
            try:
                return state.quotes[contract.contract_id]
            except KeyError:
                raise QuoteMissingError(
                    "校准会话中没有该合约的报价",
                    detail={"contract_id": contract.contract_id}) from None
        return state.quotes  # 单一目标价：适用于任意合约（向后兼容）

    def _solve(self, state: SessionState, contract: Contract,
               market: MarketParams) -> float:
        target = self._target_for(state, contract)
        vol_lo = _vol_lo_for(contract, market)

        def f(vol: float) -> float:
            return _model_price(contract, market, vol) - target

        # 障碍合约价格对波动率可能非单调（回扣与触碰概率此消彼长），
        # 端点夹逼不充分：先在对数网格上扫描符号变化，再在子区间 Brent
        # 求根。若存在多个解，取与会话行情快照波动率最接近者（确定性选择，
        # 同一会话内容必得同一解）。
        grid = [vol_lo * (_VOL_HI / vol_lo) ** (i / (_GRID_POINTS - 1))
                for i in range(_GRID_POINTS)]
        values = [f(v) for v in grid]
        for value in values:
            if not math.isfinite(value):
                raise NumericalInstabilityError(
                    "校准目标函数在扫描网格上出现非有限值",
                    detail={"quantity": "f(vol)"})
        roots: list[float] = []
        for i in range(_GRID_POINTS - 1):
            if values[i] == 0.0:
                roots.append(grid[i])
            elif values[i] * values[i + 1] < 0.0:
                roots.append(_brentq(f, grid[i], grid[i + 1], tol=self._tol,
                                     max_iter=self._max_iter))
        if values[-1] == 0.0:
            roots.append(grid[-1])
        if not roots:
            reachable = [v + target for v in values]
            raise NoSolutionError(
                "目标价格超出模型可达区间，隐含波动率不存在",
                detail={
                    "target_price": target,
                    "reachable_min": min(reachable),
                    "reachable_max": max(reachable),
                    "vol_bounds": [vol_lo, _VOL_HI],
                })
        return min(roots, key=lambda v: abs(v - market.volatility))

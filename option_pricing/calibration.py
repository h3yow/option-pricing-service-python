"""校准会话：由目标市价反解隐含波动率，并按行情参数快照缓存复用。

约定：
- 隐含波动率的求解目标函数随合约类型选择：欧式/亚式用 BS 解析式
  （市场惯例，且亚式的 MC 目标函数噪声大、不适合求根）；单障碍合约
  必须用 Reiner-Rubinstein 障碍解析式——障碍报价相对 vanilla 价格
  没有可比性，用 BS 反解会把合法的障碍报价误判为"无解"。
- 缓存键为 (contract, market_params) 整体；行情参数任何字段变化都会得到
  新键，从而强制重新校准，绝不沿用旧结果。
- 并发同一键采用 single-flight：只有一个线程真正求解，其余等待共享结果。
- 缓存为容量有界的 LRU，资源占用可预期。
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict

from .contracts import Contract, ExerciseStyle, MarketParams
from .errors import (NoConvergenceError, NoSolutionError,
                     NumericalInstabilityError)
from .pricing.barrier import barrier_price
from .pricing.black_scholes import bs_price

_VOL_LO, _VOL_HI = 1e-4, 5.0


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


class _InFlight:
    __slots__ = ("event", "result", "error")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: float | None = None
        self.error: Exception | None = None


class CalibrationSession:
    """一次校准会话。线程安全；缓存容量有界（LRU）。

    目标价默认是单一内置值；子类可重写 _target_for 实现按合约取价
    （如按客户会话各自的一组报价）。
    """

    def __init__(self, target_price: float | None, *, max_entries: int = 128,
                 max_iter: int = 100, tol: float = 1e-10) -> None:
        if target_price is not None and (
                not math.isfinite(target_price) or target_price <= 0.0):
            from .errors import ValidationError
            raise ValidationError(
                "校准目标价格必须为正有限值",
                detail={"target_price": target_price})
        self._target = target_price
        self._max_entries = max_entries
        self._max_iter = max_iter
        self._tol = tol
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple[Contract, MarketParams], float] = OrderedDict()
        self._inflight: dict[tuple[Contract, MarketParams], _InFlight] = {}

    def _target_for(self, contract: Contract) -> float:
        """返回该合约的校准目标价。默认单一目标；子类可按合约覆盖。"""
        assert self._target is not None
        return self._target

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    def calibrated_volatility(self, contract: Contract,
                              market: MarketParams) -> float:
        """返回该合约在当前行情下的隐含波动率；失败时抛出可区分错误。"""
        key = (contract, market)
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
            vol = self._solve(contract, market)
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

    def _solve(self, contract: Contract, market: MarketParams) -> float:
        target = self._target_for(contract)

        if contract.style is ExerciseStyle.BARRIER:
            def model_price(vol: float) -> float:
                return barrier_price(
                    contract.option_type, contract.barrier_direction,
                    contract.knock, market.spot, contract.strike,
                    contract.barrier, contract.maturity_years,
                    market.risk_free_rate, vol, contract.rebate)
        else:
            def model_price(vol: float) -> float:
                return bs_price(contract.option_type, market.spot,
                                contract.strike, contract.maturity_years,
                                market.risk_free_rate, vol)

        def f(vol: float) -> float:
            return model_price(vol) - target

        # 障碍解析式在极低波动率下反射因子溢出，求根区间下界收紧到 1%
        lo = 1e-2 if contract.style is ExerciseStyle.BARRIER else _VOL_LO
        hi = _VOL_HI
        # 障碍价格对波动率可能非单调（如 U 形），先几何扫描找异号子区间，
        # 再在该子区间上 Brent 求根；全程同号才判定无解。
        bracket = self._find_bracket(f, lo, hi)
        if bracket is None:
            raise NoSolutionError(
                "目标价格在波动率扫描区间内不可 bracket，隐含波动率不存在",
                detail={
                    "target_price": target,
                    "price_at_vol_lo": f(lo) + target,
                    "price_at_vol_hi": f(hi) + target,
                    "vol_bounds": [lo, hi],
                })
        return _brentq(f, bracket[0], bracket[1], tol=self._tol,
                       max_iter=self._max_iter)

    @staticmethod
    def _find_bracket(f, lo: float, hi: float,
                      samples: int = 32) -> tuple[float, float] | None:
        """几何格点扫描，返回首个异号子区间；端点恰为根时直接返回退化区间。"""
        prev_x, prev_f = lo, f(lo)
        if prev_f == 0.0:
            return (lo, lo)
        ratio = (hi / lo) ** (1.0 / samples)
        for i in range(1, samples + 1):
            x = lo * ratio ** i
            fx = f(x)
            if fx == 0.0:
                return (x, x)
            if (prev_f > 0.0) != (fx > 0.0):
                return (prev_x, x)
            prev_x, prev_f = x, fx
        return None

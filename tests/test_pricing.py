"""定价引擎单测：日志打印输入参数与判定依据。"""

import logging
import math

import pytest

from option_pricing.contracts import OptionType
from option_pricing.errors import NumericalInstabilityError
from option_pricing.pricing.asian import asian_greeks, asian_price
from option_pricing.pricing.black_scholes import bs_greeks, bs_price

log = logging.getLogger(__name__)

S, K, T, R, V = 100.0, 100.0, 1.0, 0.05, 0.2


def test_bs_textbook_values():
    call = bs_price(OptionType.CALL, S, K, T, R, V)
    put = bs_price(OptionType.PUT, S, K, T, R, V)
    log.info("输入 S=%s K=%s T=%s r=%s sigma=%s -> call=%.6f put=%.6f;"
             "判定依据: 教科书基准 call=10.4506 put=5.5735", S, K, T, R, V,
             call, put)
    assert call == pytest.approx(10.4506, abs=1e-3)
    assert put == pytest.approx(5.5735, abs=1e-3)


def test_bs_put_call_parity():
    call = bs_price(OptionType.CALL, S, K, T, R, V)
    put = bs_price(OptionType.PUT, S, K, T, R, V)
    lhs, rhs = call - put, S - K * math.exp(-R * T)
    log.info("平价判定: C-P=%.10f vs S-Ke^-rT=%.10f", lhs, rhs)
    assert lhs == pytest.approx(rhs, abs=1e-9)


def test_bs_greeks_ranges():
    g = bs_greeks(OptionType.CALL, S, K, T, R, V)
    log.info("希腊值 %s; 判定依据: 0<delta<1, gamma>0, vega>0, theta<0", g)
    assert 0.0 < g.delta < 1.0
    assert g.gamma > 0.0 and g.vega > 0.0 and g.theta < 0.0


def test_asian_deterministic_same_seed():
    p1, e1 = asian_price(OptionType.CALL, S, K, T, R, V, 12, seed=7)
    p2, e2 = asian_price(OptionType.CALL, S, K, T, R, V, 12, seed=7)
    log.info("两次同 seed 定价 %.12f / %.12f; 判定依据: 必须逐位一致", p1, p2)
    assert p1 == p2 and e1 == e2


def test_asian_below_european_and_stderr_bounded():
    ap, err = asian_price(OptionType.CALL, S, K, T, R, V, 12, seed=7)
    ep = bs_price(OptionType.CALL, S, K, T, R, V)
    log.info("亚式 %.4f (stderr %.4f) vs 欧式 %.4f; 判定依据: 平均降低"
             "波动, 亚式平值看涨应低于欧式且相对标准误 <1%%", ap, err, ep)
    assert ap < ep
    assert err / ap < 0.01


def test_asian_greeks_sane():
    g = asian_greeks(OptionType.CALL, S, K, T, R, V, 12, seed=7)
    log.info("亚式希腊值 %s; 判定依据: 0<delta<1, gamma>0", g)
    assert 0.0 < g.delta < 1.0 and g.gamma > 0.0

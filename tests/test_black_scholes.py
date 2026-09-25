import logging

import pytest

from app.errors import ErrorCode, PricingError
from app.pricing.black_scholes import bs_greeks, bs_price, implied_vol

PARAMS = dict(S=100.0, K=100.0, T=1.0, r=0.05, q=0.02, sigma=0.25)


def test_parity_call_put():
    c = bs_price("call", **PARAMS)
    p = bs_price("put", **PARAMS)
    forward = PARAMS["S"] * pow(2.718281828459045, -0.02) - PARAMS["K"] * pow(2.718281828459045, -0.05)
    assert c - p == pytest.approx(forward, abs=1e-9)


@pytest.mark.parametrize("otype", ["call", "put"])
def test_greeks_match_finite_differences(otype, caplog):
    caplog.set_level(logging.INFO)
    logging.getLogger("tests").info(
        "FD greek check option=%s params=%s", otype, PARAMS
    )
    base = bs_price(otype, **PARAMS)
    g = bs_greeks(otype, **PARAMS)

    h = 1e-4
    bumped = {**PARAMS}
    bumped["S"] += h
    d_up = bs_price(otype, **bumped)
    bumped["S"] -= 2 * h
    d_dn = bs_price(otype, **bumped)
    assert g["delta"] == pytest.approx((d_up - d_dn) / (2 * h), abs=1e-5)
    assert g["gamma"] == pytest.approx((d_up - 2 * base + d_dn) / h ** 2, abs=1e-4)

    bumped = {**PARAMS}
    bumped["sigma"] += h
    v_up = bs_price(otype, **bumped)
    bumped["sigma"] -= 2 * h
    v_dn = bs_price(otype, **bumped)
    assert g["vega"] == pytest.approx((v_up - v_dn) / (2 * h), abs=1e-5)

    bumped = {**PARAMS}
    bumped["r"] += h
    r_up = bs_price(otype, **bumped)
    bumped["r"] -= 2 * h
    r_dn = bs_price(otype, **bumped)
    assert g["rho"] == pytest.approx((r_up - r_dn) / (2 * h), abs=1e-5)

    bumped = {**PARAMS}
    bumped["T"] += h
    t_up = bs_price(otype, **bumped)
    bumped["T"] -= 2 * h
    t_dn = bs_price(otype, **bumped)
    # theta is dV/dt (calendar time) = -dV/dT
    assert g["theta"] == pytest.approx(-(t_up - t_dn) / (2 * h), abs=1e-4)


@pytest.mark.parametrize("otype", ["call", "put"])
def test_implied_vol_roundtrip(otype):
    price = bs_price(otype, **PARAMS)
    iv = implied_vol(
        otype, price,
        S=PARAMS['S'], K=PARAMS['K'], T=PARAMS['T'],
        r=PARAMS['r'], q=PARAMS['q'],
    )
    assert iv == pytest.approx(0.25, abs=1e-7)


def test_implied_vol_below_intrinsic_rejected():
    with pytest.raises(PricingError) as exc:
        implied_vol("call", 0.01, S=100.0, K=90.0, T=1.0, r=0.05, q=0.0)
    assert exc.value.code is ErrorCode.CALIBRATION_NO_SOLUTION
    assert "intrinsic" in exc.value.message


def test_implied_vol_nonpositive_premium_rejected():
    with pytest.raises(PricingError) as exc:
        implied_vol("put", 0.0, S=100.0, K=100.0, T=1.0, r=0.05, q=0.0)
    assert exc.value.code is ErrorCode.CALIBRATION_NO_SOLUTION

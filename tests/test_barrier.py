"""Barrier option tests.

Branches are exercised via in-out parity, zero-payoff degeneracies and the
barrier-to-vanilla limits; the formulas themselves were additionally
validated against Brownian-bridge Monte Carlo during development.
"""
import logging

import pytest

from app.errors import ErrorCode, PricingError
from app.pricing.barrier import barrier_price
from app.pricing.black_scholes import bs_price

T, R, Q, SIG = 0.5, 0.08, 0.04, 0.25
BT = ["down_and_in", "down_and_out", "up_and_in", "up_and_out"]


@pytest.mark.parametrize("ot", ["call", "put"])
@pytest.mark.parametrize("direction", ["down", "up"])
def test_in_out_parity(ot, direction, caplog):
    caplog.set_level(logging.INFO)
    S, K = 100.0, 100.0
    H = 95.0 if direction == "down" else 120.0
    pin = barrier_price(f"{direction}_and_in", ot, S, K, H, 0.0, T, R, Q, SIG)
    pout = barrier_price(f"{direction}_and_out", ot, S, K, H, 0.0, T, R, Q, SIG)
    vanilla = bs_price(ot, S, K, T, R, Q, SIG)
    logging.getLogger("tests").info(
        "parity ot=%s dir=%s S=%s K=%s H=%s in=%.6f out=%.6f vanilla=%.6f",
        ot, direction, S, K, H, pin, pout, vanilla,
    )
    assert pin + pout == pytest.approx(vanilla, abs=1e-9)


@pytest.mark.parametrize("ot", ["call", "put"])
def test_deep_far_barrier_recovers_vanilla(ot):
    # A barrier that cannot realistically be hit -> knock-out ~ vanilla,
    # knock-in ~ 0.
    S = 100.0
    K = 100.0
    pin = barrier_price("up_and_in", ot, S, K, 1000.0, 0.0, T, R, Q, SIG)
    pout = barrier_price("up_and_out", ot, S, K, 1000.0, 0.0, T, R, Q, SIG)
    assert pin == pytest.approx(0.0, abs=1e-9)
    assert pout == pytest.approx(bs_price(ot, S, K, T, R, Q, SIG), abs=1e-9)


def test_zero_payoff_degeneracies():
    # Up-and-out call with barrier below strike (H<K) can never pay off.
    assert barrier_price("up_and_out", "call", 100, 130, 120, 0.0, T, R, Q, SIG) == 0.0
    # Down-and-out put with barrier above strike (K<=H) can never pay off.
    assert barrier_price("down_and_out", "put", 100, 90, 95, 0.0, T, R, Q, SIG) == 0.0


@pytest.mark.parametrize("direction", ["down", "up"])
def test_rebate_nonnegative_and_monotone(direction):
    S, K = 100.0, 100.0
    H = 95.0 if direction == "down" else 120.0
    p0 = barrier_price(f"{direction}_and_out", "call", S, K, H, 0.0, T, R, Q, SIG)
    p1 = barrier_price(f"{direction}_and_out", "call", S, K, H, 1.0, T, R, Q, SIG)
    p5 = barrier_price(f"{direction}_and_out", "call", S, K, H, 5.0, T, R, Q, SIG)
    # Knock-out rebate is paid at touch: value is linear in the rebate.
    assert p1 - p0 == pytest.approx((p5 - p0) / 5.0, abs=1e-9)
    assert p1 > p0


def test_knock_in_rebate_paid_at_expiry_when_not_hit():
    # Deep far up barrier for an up-and-in: never knocks in, so value is the
    # expiry-discounted rebate (plus negligible option value).
    p = barrier_price("up_and_in", "put", 100, 130, 1000.0, 2.0, T, R, Q, SIG)
    assert p == pytest.approx(2.0 * pow(2.718281828459045, -R * T), abs=1e-8)


def test_nonfinite_input_is_instability():
    with pytest.raises(PricingError) as exc:
        barrier_price("up_and_in", "call", 100, 100, float("inf"), 0, T, R, Q, SIG)
    assert exc.value.code is ErrorCode.NUMERICAL_INSTABILITY

"""Brownian-bridge Monte Carlo cross-check for all barrier branches.

Skipped if numpy is unavailable (it is a dev-only dependency).
"""
import math

import pytest

np = pytest.importorskip("numpy")

from app.pricing.barrier import barrier_price

T, R, Q, SIG = 0.5, 0.08, 0.04, 0.25


def _mc(bt, ot, S, K, H, rebate, n_paths=220_000, n_steps=240, seed=7):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    inc = (R - Q - 0.5 * SIG ** 2) * dt + SIG * np.sqrt(dt) * z
    lp = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(inc, axis=1)], axis=1
    )
    up, dn = lp[:, 1:], lp[:, :-1]
    lh = math.log(H / S)
    if bt.startswith("up"):
        pc = np.where(
            (up < lh) & (dn < lh),
            np.exp(-2 * (lh - up) * (lh - dn) / (SIG ** 2 * dt)),
            0.0,
        )
        hit_mat = (up >= lh) | (rng.random(pc.shape) < pc)
    else:
        pc = np.where(
            (up > lh) & (dn > lh),
            np.exp(-2 * (up - lh) * (dn - lh) / (SIG ** 2 * dt)),
            0.0,
        )
        hit_mat = (up <= lh) | (rng.random(pc.shape) < pc)
    hit = np.any(hit_mat, axis=1)
    first = np.argmax(hit_mat, axis=1)
    ht = np.where(hit, (first + 1) * dt, 0.0)
    st = S * np.exp(lp[:, -1])
    alive = hit if bt.endswith("in") else ~hit
    payoff = np.maximum(st - K, 0) if ot == "call" else np.maximum(K - st, 0)
    v = math.exp(-R * T) * np.where(alive, payoff, 0)
    if rebate > 0:
        if bt.endswith("in"):
            v += math.exp(-R * T) * np.where(~hit, rebate, 0)
        else:
            v += np.where(hit, rebate * np.exp(-R * ht), 0)
    return float(v.mean())


CASES = [
    ("down_and_in", "call", 100, 90, 95, 0),
    ("down_and_out", "call", 100, 90, 95, 0),
    ("down_and_in", "call", 100, 105, 95, 0),
    ("down_and_out", "call", 100, 105, 95, 0),
    ("up_and_in", "call", 100, 100, 120, 0),
    ("up_and_out", "call", 100, 100, 120, 0),
    ("up_and_in", "call", 100, 130, 120, 0),
    ("up_and_out", "call", 100, 130, 120, 0),
    ("down_and_in", "put", 100, 105, 95, 0),
    ("down_and_out", "put", 100, 105, 95, 0),
    ("down_and_in", "put", 100, 90, 95, 0),
    ("down_and_out", "put", 100, 90, 95, 0),
    ("up_and_in", "put", 100, 90, 120, 0),
    ("up_and_out", "put", 100, 90, 120, 0),
    ("up_and_in", "put", 100, 130, 120, 0),
    ("up_and_out", "put", 100, 130, 120, 0),
    ("down_and_out", "call", 100, 100, 95, 3),
    ("up_and_in", "put", 100, 100, 120, 2),
    ("up_and_out", "put", 100, 130, 120, 2),
    ("down_and_in", "call", 100, 105, 95, 3),
]


@pytest.mark.parametrize("bt,ot,S,K,H,rebate", CASES)
def test_barrier_branch_vs_mc(bt, ot, S, K, H, rebate):
    analytic = barrier_price(bt, ot, S, K, H, rebate, T, R, Q, SIG)
    mc = _mc(bt, ot, S, K, H, rebate)
    # generous tolerance: MC error plus bridge timing discretization
    assert analytic == pytest.approx(mc, abs=0.13)

import math

import pytest

from app.errors import ErrorCode, PricingError
from app.numerics import brentq, norm_cdf, norm_pdf


def test_norm_known_values():
    assert norm_cdf(0.0) == pytest.approx(0.5, abs=1e-12)
    assert norm_cdf(1.96) == pytest.approx(0.975, abs=1e-4)
    assert norm_pdf(0.0) == pytest.approx(1 / math.sqrt(2 * math.pi), abs=1e-12)


def test_brentq_finds_roots():
    assert brentq(lambda x: x ** 2 - 2, 0.0, 4.0) == pytest.approx(math.sqrt(2), abs=1e-8)
    assert brentq(math.sin, 2.0, 4.0) == pytest.approx(math.pi, abs=1e-8)


def test_brentq_no_sign_change_is_no_solution():
    with pytest.raises(PricingError) as exc:
        brentq(lambda x: x ** 2 + 1.0, -1.0, 1.0)
    assert exc.value.code is ErrorCode.CALIBRATION_NO_SOLUTION


def test_brentq_nonfinite_is_instability():
    with pytest.raises(PricingError) as exc:
        brentq(lambda x: float("nan"), 0.0, 1.0)
    assert exc.value.code is ErrorCode.NUMERICAL_INSTABILITY


def test_norm_nonfinite_is_instability():
    with pytest.raises(PricingError) as exc:
        norm_cdf(float("inf"))
    assert exc.value.code is ErrorCode.NUMERICAL_INSTABILITY

import logging
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.pricing.black_scholes import bs_price

AS_OF = "2026-09-25"
EXPIRY = "2027-09-25"


@pytest.fixture()
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


def euro_item(cid="c1", strike=100.0, expiry=EXPIRY, sigma=0.25):
    return {
        "contract": {"contract_id": cid, "style": "european",
                     "option_type": "call", "strike": strike, "expiry": expiry},
        "model": "black_scholes", "sigma_override": sigma,
    }


def barrier_item():
    return {
        "contract": {"contract_id": "b1", "style": "barrier",
                     "option_type": "call", "strike": 100.0, "expiry": EXPIRY,
                     "barrier": 120.0, "barrier_type": "up_and_out",
                     "rebate": 2.0},
        "model": "reiner_rubinstein", "sigma_override": 0.25,
    }


def test_health_and_catalog(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "black_scholes" in body["supported_models"]
    cat = client.get("/v1/catalog").json()
    assert cat["builtin_market"]["spot"] == 100.0


def test_single_and_batch_identical(client, caplog):
    caplog.set_level(logging.INFO)
    r1 = client.post("/v1/price", json={"items": [euro_item()]})
    assert r1.status_code == 200, r1.text
    single = r1.json()
    rb = client.post("/v1/price/batch",
                     json={"items": [euro_item(), barrier_item()]})
    assert rb.status_code == 200
    batch = rb.json()["results"]
    logging.getLogger("tests").info(
        "single=%s batch0=%s greeks_present=%s",
        single["price"], batch[0]["result"]["price"],
        "vega" in single["greeks"],
    )
    assert batch[0]["ok"] and batch[1]["ok"]
    assert batch[0]["result"]["price"] == single["price"]
    assert batch[0]["result"]["greeks"] == single["greeks"]
    # analytic expectation
    assert single["price"] == pytest.approx(
        bs_price("call", 100, 100, 1.0, 0.05, 0.02, 0.25), abs=1e-9
    )


def test_batch_classifies_invalid_contracts(client):
    expired = euro_item(expiry="2020-01-01")
    wrong_model = {
        "contract": {"style": "european", "option_type": "call",
                     "strike": 100.0, "expiry": EXPIRY},
        "model": "reiner_rubinstein", "sigma_override": 0.25,
    }
    bad_sigma = euro_item(sigma=10.0)
    r = client.post("/v1/price/batch",
                    json={"items": [euro_item(), expired, wrong_model, bad_sigma]})
    assert r.status_code == 200
    results = r.json()["results"]
    codes = [x.get("error_code") for x in results]
    assert codes == [None, "CONTRACT_EXPIRED", "MODEL_NOT_APPLICABLE",
                     "INVALID_PARAMETER"]


def test_single_expired_returns_distinct_code(client):
    r = client.post("/v1/price", json={"items": [euro_item(expiry="2020-01-01")]})
    assert r.status_code == 422
    assert r.json()["error_code"] == "CONTRACT_EXPIRED"


def test_session_lifecycle_and_recalibration(client):
    premium = round(bs_price("call", 100, 100, 1.0, 0.05, 0.02, 0.3), 6)
    payload = {
        "market": {"spot": 100.0, "risk_free_rate": 0.05,
                   "dividend_yield": 0.02},
        "quotes": [{"strike": 100.0, "expiry": EXPIRY,
                    "option_type": "call", "premium": premium}],
    }
    r = client.post("/v1/sessions", json=payload)
    assert r.status_code == 201, r.text
    sid = r.json()["session_id"]
    assert r.json()["calibrated_sigma"] == pytest.approx(0.3, abs=1e-4)

    # price via session without sigma
    item = {
        "contract": {"contract_id": "sc", "style": "european",
                     "option_type": "call", "strike": 100.0, "expiry": EXPIRY},
        "model": "black_scholes",
    }
    rp = client.post("/v1/price", json={"session_id": sid, "items": [item]})
    assert rp.json()["calibrated"] is True

    # change rate -> recalibration must use new rate; price differs
    client.patch(f"/v1/sessions/{sid}/market",
                 json={"risk_free_rate": 0.09})
    rp2 = client.post("/v1/price", json={"session_id": sid, "items": [item]})
    assert rp2.json()["sigma_used"] != rp.json()["sigma_used"]

    client.delete(f"/v1/sessions/{sid}")
    assert client.get(f"/v1/sessions/{sid}").status_code == 404


def test_unknown_session_is_404(client):
    item = euro_item()
    item.pop("sigma_override")
    r = client.post("/v1/price",
                    json={"session_id": "ghost", "items": [item]})
    assert r.status_code == 404
    assert r.json()["error_code"] == "SESSION_NOT_FOUND"


def test_no_sigma_without_session_rejected(client):
    item = euro_item()
    item.pop("sigma_override")
    r = client.post("/v1/price", json={"items": [item]})
    assert r.status_code == 422
    assert r.json()["error_code"] == "INVALID_PARAMETER"


def test_empty_batch_rejected(client):
    r = client.post("/v1/price/batch", json={"items": []})
    assert r.status_code == 422
    assert r.json()["error_code"] == "EMPTY_BATCH"


def test_concurrent_requests_consistent(client):
    item = euro_item()

    def call():
        r = client.post("/v1/price/batch", json={"items": [item, item, item]})
        return r.json()

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: call(), range(16)))
    prices = {
        resp["results"][0]["result"]["price"]
        for resp in responses
    }
    assert len(prices) == 1

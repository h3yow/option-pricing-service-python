"""接口层测试：端点行为、行情更新触发重校准、并发请求一致。"""

import logging
import threading

import pytest
from fastapi.testclient import TestClient

from main import app

log = logging.getLogger(__name__)


@pytest.fixture()
def client():
    c = TestClient(app, raise_server_exceptions=False)
    yield c
    c.post("/market", json={"spot": 100.0, "volatility": 0.25,
                            "risk_free_rate": 0.05})


def test_single_and_batch_endpoints_agree(client):
    single = client.post("/price", json={"contract_id": "EU-CALL-100"})
    assert single.status_code == 200
    batch = client.post("/price/batch", json={"items": [
        {"contract_id": "EU-CALL-100"},
        {"contract_id": "ASIAN-CALL-100"},
        {"contract_id": "EU-CALL-100"},
    ]})
    assert batch.status_code == 200
    items = batch.json()
    log.info("单笔=%s 批量=%s; 判定依据: 两入口结论逐位一致",
             single.json(), items)
    assert items[0] == single.json() == items[2]


def test_error_taxonomy_over_http(client):
    cases = [
        ({"contract_id": "NOPE"}, 404, "unknown_contract"),
        ({"contract_id": "EXPIRED-CALL"}, 410, "contract_expired"),
        ({"contract_id": "ASIAN-CALL-100", "model": "black_scholes"},
         400, "model_not_applicable"),
    ]
    for payload, status, code in cases:
        r = client.post("/price", json=payload)
        body = r.json()
        got = (body.get("error") or body.get("detail", {}).get("error"))["code"]
        log.info("请求 %s -> HTTP %d code=%s; 判定依据: 预期 %d/%s",
                 payload, r.status_code, got, status, code)
        assert r.status_code == status and got == code


def test_market_update_triggers_recalibration(client):
    before = client.post("/price", json={"contract_id": "EU-CALL-100"}).json()
    r = client.post("/market", json={"spot": 105.0})
    assert r.status_code == 200
    after = client.post("/price", json={"contract_id": "EU-CALL-100"}).json()
    log.info("spot 100->105: iv %s -> %s; 判定依据: 参数变更必须重校准",
             before["calibrated_volatility"], after["calibrated_volatility"])
    assert after["calibrated_volatility"] != before["calibrated_volatility"]


def test_unreachable_market_update_rejected_atomically(client):
    r = client.post("/market", json={"spot": 110.0})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "no_solution"
    spot = client.get("/market").json()["spot"]
    log.info("不可达更新被拒后 spot=%s; 判定依据: 应保持旧值 100.0", spot)
    assert spot == 100.0


def test_concurrent_http_requests_consistent(client):
    results: list = [None] * 12

    def work(i):
        r = client.post("/price", json={
            "contract_id": "EU-CALL-100" if i % 2 == 0 else "ASIAN-CALL-100"})
        assert r.status_code == 200
        results[i] = r.json()

    threads = [threading.Thread(target=work, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    eu = {repr(results[i]) for i in range(0, 12, 2)}
    az = {repr(results[i]) for i in range(1, 12, 2)}
    log.info("12 并发请求: EU 唯一结果数=%d AZ 唯一结果数=%d; 判定依据: 各为 1",
             len(eu), len(az))
    assert len(eu) == 1 and len(az) == 1


# ---------------------------------------------------------------- 会话端点

@pytest.fixture()
def session_id(client):
    r = client.post("/sessions", json={
        "market": {"spot": 100.0, "volatility": 0.25, "risk_free_rate": 0.05},
        "quotes": {"EU-CALL-100": 12.336, "UO-CALL-120": 3.0}})
    assert r.status_code == 201
    sid = r.json()["session_id"]
    yield sid
    client.delete(f"/sessions/{sid}")


def test_session_lifecycle_over_http(client, session_id):
    r = client.get(f"/sessions/{session_id}")
    assert r.status_code == 200 and r.json()["version"] == 1
    r = client.put(f"/sessions/{session_id}",
                   json={"quotes": {"EU-CALL-100": 13.0, "UO-CALL-120": 3.5}})
    assert r.status_code == 200 and r.json()["version"] == 2
    r = client.delete(f"/sessions/{session_id}")
    assert r.status_code == 204
    r = client.get(f"/sessions/{session_id}")
    log.info("删除后 GET -> %d %s; 判定依据: 404 session_not_found",
             r.status_code, r.json())
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "session_not_found"


def test_session_price_carries_session_identity(client, session_id):
    r = client.post(f"/sessions/{session_id}/price",
                    json={"contract_id": "UO-CALL-120"})
    j = r.json()
    log.info("会话定价: price=%.4f model=%s sid=%s v%d; 判定依据: 返回"
             "必须标明出自哪个会话的哪一版", j["price"], j["model"],
             j["session_id"], j["session_version"])
    assert r.status_code == 200
    assert j["session_id"] == session_id and j["session_version"] == 1
    assert j["model"] == "barrier_analytic"


def test_session_update_recalibrates(client, session_id):
    before = client.post(f"/sessions/{session_id}/price",
                         json={"contract_id": "EU-CALL-100"}).json()
    client.put(f"/sessions/{session_id}",
               json={"quotes": {"EU-CALL-100": 13.5, "UO-CALL-120": 3.5}})
    after = client.post(f"/sessions/{session_id}/price",
                        json={"contract_id": "EU-CALL-100"}).json()
    log.info("更新报价 12.336->13.5: iv %.4f -> %.4f, v%d -> v%d; 判定依据:"
             " 更新后必须重新校准", before["calibrated_volatility"],
             after["calibrated_volatility"], before["session_version"],
             after["session_version"])
    assert after["session_version"] == 2
    assert after["calibrated_volatility"] != before["calibrated_volatility"]


def test_session_batch_and_single_agree(client, session_id):
    single = client.post(f"/sessions/{session_id}/price",
                         json={"contract_id": "EU-CALL-100"}).json()
    batch = client.post(f"/sessions/{session_id}/price/batch", json={
        "items": [{"contract_id": "EU-CALL-100"},
                  {"contract_id": "UO-CALL-120"},
                  {"contract_id": "EU-CALL-100"}]}).json()
    log.info("会话内单笔 vs 批量; 判定依据: 同一合约两入口逐位一致")
    assert batch[0] == single == batch[2]


def test_session_error_taxonomy(client, session_id):
    unknown = client.post("/sessions/nope/price",
                          json={"contract_id": "EU-CALL-100"})
    no_quote = client.post(f"/sessions/{session_id}/price",
                           json={"contract_id": "EU-PUT-100"})
    bad_model = client.post(f"/sessions/{session_id}/price",
                            json={"contract_id": "UO-CALL-120",
                                  "model": "black_scholes"})
    cases = [(unknown, 404, "session_not_found"),
             (no_quote, 422, "quote_not_found"),
             (bad_model, 400, "model_not_applicable")]
    for r, status, code in cases:
        got = r.json()["error"]["code"]
        log.info("%s -> HTTP %d code=%s; 判定依据: 预期 %d/%s",
                 r.request.url.path, r.status_code, got, status, code)
        assert r.status_code == status and got == code

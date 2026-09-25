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

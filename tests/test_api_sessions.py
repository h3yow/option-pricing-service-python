"""接口层会话测试：HTTP 生命周期、错误码映射、批量复用与并发一致。"""

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
    # 清理本用例创建的会话，避免影响其它用例（容量上限）
    for s in c.get("/sessions").json():
        c.delete(f"/sessions/{s['session_id']}")


def _create(client, quotes=None, client_name="clientA"):
    payload = {
        "client": client_name,
        "underlying": "XYZ",
        "quotes": quotes or {"EU-CALL-100": 12.336},
        "market": {"spot": 100.0, "volatility": 0.25, "risk_free_rate": 0.05},
    }
    r = client.post("/sessions", json=payload)
    assert r.status_code == 201, r.json()
    return r.json()


def test_session_lifecycle_over_http(client):
    created = _create(client)
    sid = created["session_id"]
    log.info("创建会话 %s version=%d; 判定依据: 可查询/更新/删除",
             sid, created["version"])
    got = client.get(f"/sessions/{sid}")
    assert got.status_code == 200 and got.json()["quotes"] == {
        "EU-CALL-100": 12.336}
    assert any(s["session_id"] == sid for s in client.get("/sessions").json())

    updated = client.post(f"/sessions/{sid}",
                          json={"quotes": {"EU-CALL-100": 14.0}})
    assert updated.status_code == 200 and updated.json()["version"] == 1

    assert client.delete(f"/sessions/{sid}").status_code == 200
    gone = client.get(f"/sessions/{sid}")
    assert gone.status_code == 404
    assert gone.json()["error"]["code"] == "unknown_session"


def test_price_in_session_carries_provenance(client):
    sid = _create(client)["session_id"]
    r = client.post("/price",
                    json={"contract_id": "EU-CALL-100", "session_id": sid})
    body = r.json()
    log.info("会话内定价: session_id=%s version=%s iv=%.6f; 判定依据: "
             "返回必须携带会话归属与版本", body["session_id"],
             body["session_version"], body["calibrated_volatility"])
    assert r.status_code == 200
    assert body["session_id"] == sid and body["session_version"] == 0

    client.post(f"/sessions/{sid}", json={"quotes": {"EU-CALL-100": 14.0}})
    r2 = client.post("/price",
                     json={"contract_id": "EU-CALL-100", "session_id": sid})
    log.info("更新后 iv %.6f -> %.6f, version=%d; 判定依据: 重校准且版本+1",
             body["calibrated_volatility"],
             r2.json()["calibrated_volatility"],
             r2.json()["session_version"])
    assert r2.json()["session_version"] == 1
    assert r2.json()["calibrated_volatility"] != body["calibrated_volatility"]


def test_session_error_taxonomy_over_http(client):
    sid = _create(client)["session_id"]
    cases = [
        ({"contract_id": "EU-CALL-100", "session_id": "S9999"},
         404, "unknown_session"),
        ({"contract_id": "ASIAN-CALL-100", "session_id": sid},
         422, "quote_missing"),
    ]
    for payload, status, code in cases:
        r = client.post("/price", json=payload)
        got = r.json()["error"]["code"]
        log.info("请求 %s -> HTTP %d code=%s; 判定依据: 预期 %d/%s",
                 payload, r.status_code, got, status, code)
        assert r.status_code == status and got == code
    # 不可达报价更新 -> 422 no_solution, 且状态不变
    r = client.post(f"/sessions/{sid}", json={"quotes": {"EU-CALL-100": 1.0}})
    assert r.status_code == 422 and r.json()["error"]["code"] == "no_solution"
    assert client.get(f"/sessions/{sid}").json()["version"] == 0


def test_session_capacity_over_http(client):
    created = [_create(client, client_name=f"c{i}") for i in range(64)]
    r = client.post("/sessions", json={
        "client": "overflow", "underlying": "XYZ",
        "quotes": {"EU-CALL-100": 12.336},
        "market": {"spot": 100.0, "volatility": 0.25, "risk_free_rate": 0.05}})
    got = r.json()["error"]
    log.info("第 65 个会话 -> HTTP %d code=%s resource=%s; 判定依据: "
             "429/capacity_exceeded/sessions", r.status_code, got["code"],
             got["detail"].get("resource"))
    assert r.status_code == 429 and got["code"] == "capacity_exceeded"
    assert got["detail"]["resource"] == "sessions"
    for s in created:
        client.delete(f"/sessions/{s['session_id']}")


def test_batch_mixed_sessions_consistent_with_single(client):
    a = _create(client, client_name="A")["session_id"]
    b = _create(client, quotes={"EU-CALL-100": 10.0}, client_name="B")[
        "session_id"]
    single_a = client.post("/price", json={
        "contract_id": "EU-CALL-100", "session_id": a}).json()
    single_builtin = client.post("/price",
                                 json={"contract_id": "EU-CALL-100"}).json()
    batch = client.post("/price/batch", json={"items": [
        {"contract_id": "EU-CALL-100", "session_id": a},
        {"contract_id": "EU-CALL-100", "session_id": b},
        {"contract_id": "EU-CALL-100", "session_id": a},
        {"contract_id": "EU-CALL-100"},
    ]})
    assert batch.status_code == 200
    items = batch.json()
    log.info("批量混合会话: ivs=%s; 判定依据: 同会话同结果且与单笔一致, "
             "跨会话不复用校准", [i["calibrated_volatility"] for i in items])
    assert items[0] == single_a == items[2]
    assert items[3] == single_builtin
    assert items[1]["calibrated_volatility"] != items[0][
        "calibrated_volatility"]


def test_concurrent_http_requests_across_sessions(client):
    a = _create(client, client_name="A")["session_id"]
    b = _create(client, quotes={"EU-CALL-100": 10.0}, client_name="B")[
        "session_id"]
    results: list = [None] * 12

    def work(i):
        sid = a if i % 2 == 0 else b
        r = client.post("/price", json={"contract_id": "EU-CALL-100",
                                        "session_id": sid})
        assert r.status_code == 200
        results[i] = r.json()

    threads = [threading.Thread(target=work, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ra = {repr(results[i]) for i in range(0, 12, 2)}
    rb = {repr(results[i]) for i in range(1, 12, 2)}
    log.info("12 并发跨会话请求: A 唯一结果=%d B 唯一结果=%d; 判定依据: 各为 1",
             len(ra), len(rb))
    assert len(ra) == 1 and len(rb) == 1

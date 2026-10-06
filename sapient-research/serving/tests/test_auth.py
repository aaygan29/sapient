"""Auth: default-deny everywhere except /healthz."""
from __future__ import annotations


def test_healthz_open(client):
    assert client.get("/healthz").status_code == 200


def test_info_requires_key(client):
    assert client.get("/v1/info").status_code == 401


def test_info_rejects_bad_key(client):
    r = client.get("/v1/info", headers={"X-API-Key": "sk_sapient_not_a_real_key"})
    assert r.status_code == 401


def test_info_ok_and_honest(client, key_a):
    r = client.get("/v1/info", headers={"Authorization": f"Bearer {key_a}"})
    assert r.status_code == 200
    body = r.json()
    assert body["is_paper_model"] is False
    assert "NOT" in json_str(body["provenance"])


def json_str(d) -> str:
    import json

    return json.dumps(d)

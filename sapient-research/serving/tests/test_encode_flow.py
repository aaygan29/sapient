"""Encode -> poll -> result, plus the IP control on full parcels."""
from __future__ import annotations

import time


def _wait(client, key, job_id, timeout=5.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.get(f"/v1/jobs/{job_id}", headers={"X-API-Key": key}).json()
        if last["status"] in ("done", "error"):
            return last
        time.sleep(0.02)
    return last


def test_encode_then_poll(client, key_a):
    files = {"video": ("clip.mp4", b"\x00\x01\x02 fake video bytes", "video/mp4")}
    r = client.post("/v1/encode", headers={"X-API-Key": key_a}, files=files)
    assert r.status_code == 202
    status = _wait(client, key_a, r.json()["job_id"])
    assert status["status"] == "done"
    result = status["result"]
    assert result["roi_scores"]  # non-empty
    assert result["parcels"]["n_parcels"] == 1000
    assert result["parcels"]["values"] is None  # client A not allowed full vector
    assert result["image_data_uri"].startswith("data:image/png;base64,")
    signals = result["signals"]
    assert 0 <= signals["purchase_intent"] <= 100
    assert signals["recommendation"] in ("Buy", "Hold", "Sell")
    assert len(signals["constructs"]) == 7
    assert len(signals["timeline"]["purchase_intent"]) == result["n_timepoints"]


def test_full_parcels_only_for_allowed_client(client, key_b):
    files = {"video": ("clip.mp4", b"abc123", "video/mp4")}
    r = client.post(
        "/v1/encode",
        headers={"X-API-Key": key_b},
        files=files,
        data={"include_full_parcels": "true"},
    )
    status = _wait(client, key_b, r.json()["job_id"])
    assert status["status"] == "done"
    values = status["result"]["parcels"]["values"]
    assert values is not None and len(values) == 1000


def test_requires_some_input(client, key_a):
    r = client.post("/v1/encode", headers={"X-API-Key": key_a})
    assert r.status_code == 422

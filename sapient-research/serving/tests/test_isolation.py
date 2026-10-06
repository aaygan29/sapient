"""Tenant isolation: a client can never read another client's job (no IDOR)."""
from __future__ import annotations

import time


def _run(client, key) -> str:
    files = {"video": ("c.mp4", b"xyz", "video/mp4")}
    job_id = client.post("/v1/encode", headers={"X-API-Key": key}, files=files).json()["job_id"]
    deadline = time.time() + 5.0
    while time.time() < deadline:
        s = client.get(f"/v1/jobs/{job_id}", headers={"X-API-Key": key}).json()
        if s["status"] in ("done", "error"):
            break
        time.sleep(0.02)
    return job_id


def test_cross_tenant_read_blocked(client, key_a, key_b):
    job_a = _run(client, key_a)
    # Client B must NOT be able to read client A's job — same 404 as not-found.
    assert client.get(f"/v1/jobs/{job_a}", headers={"X-API-Key": key_b}).status_code == 404
    # Owner can read it.
    assert client.get(f"/v1/jobs/{job_a}", headers={"X-API-Key": key_a}).status_code == 200


def test_unknown_job_is_404(client, key_a):
    assert client.get("/v1/jobs/deadbeef", headers={"X-API-Key": key_a}).status_code == 404

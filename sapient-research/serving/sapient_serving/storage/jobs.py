"""In-memory job store with tenant isolation, unguessable IDs, and TTL expiry.

Security properties:
  * Job IDs are UUID4 (unguessable) — no sequential enumeration.
  * get() requires the requesting client to OWN the job, else it behaves as not-found
    (prevents cross-tenant reads / IDOR).
  * Results carry an expires_at; expired records are purged on access (data minimization).
  * Only the coarse RESULT is stored — never the raw uploaded media.

Scope note: in-memory => correct for a single instance. For multiple instances /
autoscaling, back results with Redis or an object store that enforces the same
ownership + TTL semantics.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from ..schemas import Prediction


@dataclass
class JobRecord:
    job_id: str
    client_id: str
    status: str
    created_at: float
    expires_at: Optional[float] = None
    result: Optional[Prediction] = None
    error: Optional[str] = None


class JobStore:
    def __init__(self, ttl_seconds: int) -> None:
        self.ttl = ttl_seconds
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}

    def _purge_locked(self, now: float) -> None:
        expired = [jid for jid, r in self._jobs.items() if r.expires_at and r.expires_at < now]
        for jid in expired:
            del self._jobs[jid]

    def create(self, client_id: str) -> JobRecord:
        now = time.time()
        with self._lock:
            self._purge_locked(now)
            rec = JobRecord(job_id=uuid.uuid4().hex, client_id=client_id, status="queued", created_at=now)
            self._jobs[rec.job_id] = rec
            return rec

    def set_processing(self, job_id: str) -> None:
        with self._lock:
            r = self._jobs.get(job_id)
            if r:
                r.status = "processing"

    def set_result(self, job_id: str, result: Prediction) -> None:
        with self._lock:
            r = self._jobs.get(job_id)
            if r:
                r.result = result
                r.status = "done"
                r.expires_at = time.time() + self.ttl

    def set_error(self, job_id: str, message: str) -> None:
        with self._lock:
            r = self._jobs.get(job_id)
            if r:
                r.error = message
                r.status = "error"
                r.expires_at = time.time() + self.ttl

    def get(self, job_id: str, client_id: str) -> Optional[JobRecord]:
        now = time.time()
        with self._lock:
            self._purge_locked(now)
            r = self._jobs.get(job_id)
            if r is None or r.client_id != client_id:
                return None  # tenant isolation: not yours == doesn't exist
            return r


_store: Optional[JobStore] = None


def init_store(ttl_seconds: int) -> JobStore:
    global _store
    _store = JobStore(ttl_seconds)
    return _store


def get_store() -> JobStore:
    if _store is None:
        raise RuntimeError("job store not initialized")
    return _store

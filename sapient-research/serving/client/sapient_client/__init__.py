"""Thin client for the Sapient-1 Encoding API.

This is what you hand to clients. It is pure HTTP — no model code, no weights.
Reverse-engineering the model from this package is impossible; it only knows how
to call the API with a key.

    from sapient_client import Client
    c = Client("sk_sapient_...", base_url="https://api.yourco.com")
    result = c.encode(video="ad.mp4")
    print(result["roi_scores"])
"""
from __future__ import annotations

import time
from typing import Optional, Union

import requests

__version__ = "0.1.0"


class SapientError(Exception):
    pass


def _as_bytes(x: Union[str, bytes]) -> bytes:
    if isinstance(x, bytes):
        return x
    with open(x, "rb") as fh:
        return fh.read()


class Client:
    def __init__(self, api_key: str, base_url: str = "http://localhost:8000", timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def info(self) -> dict:
        r = requests.get(f"{self.base_url}/v1/info", headers=self._headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def job(self, job_id: str) -> dict:
        r = requests.get(f"{self.base_url}/v1/jobs/{job_id}", headers=self._headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def encode(
        self,
        *,
        video: Optional[Union[str, bytes]] = None,
        audio: Optional[Union[str, bytes]] = None,
        transcript: Optional[str] = None,
        subject_idx: int = 0,
        poll: bool = True,
        poll_interval: float = 0.5,
        poll_timeout: float = 300.0,
    ) -> dict:
        files = {}
        if video is not None:
            files["video"] = ("video.mp4", _as_bytes(video), "video/mp4")
        if audio is not None:
            files["audio"] = ("audio.wav", _as_bytes(audio), "audio/wav")
        data = {"subject_idx": str(subject_idx)}
        if transcript:
            data["transcript"] = transcript

        r = requests.post(
            f"{self.base_url}/v1/encode",
            headers=self._headers,
            files=files or None,
            data=data,
            timeout=self.timeout,
        )
        r.raise_for_status()
        accepted = r.json()
        if not poll:
            return accepted

        job_id = accepted["job_id"]
        deadline = time.time() + poll_timeout
        while time.time() < deadline:
            status = self.job(job_id)
            if status["status"] == "done":
                return status["result"]
            if status["status"] == "error":
                raise SapientError(status.get("error") or "job failed")
            time.sleep(poll_interval)
        raise SapientError("timed out waiting for result")

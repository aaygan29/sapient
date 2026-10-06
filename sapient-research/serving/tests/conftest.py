"""Test harness: mock engine + a temp hashed-key store with two distinct clients.

Env is configured BEFORE importing the app so the settings singleton picks it up.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile

import pytest

_TMP = tempfile.mkdtemp(prefix="sapient-test-")
KEYS_PATH = os.path.join(_TMP, "keys.json")
KEY_A = "sk_sapient_test_AAAAAAAAAAAAAAAAAAAAAAAAAAA"
KEY_B = "sk_sapient_test_BBBBBBBBBBBBBBBBBBBBBBBBBBB"


def _h(k: str) -> str:
    return hashlib.sha256(k.encode("utf-8")).hexdigest()


with open(KEYS_PATH, "w", encoding="utf-8") as _fh:
    json.dump(
        {
            _h(KEY_A): {"client_id": "client-a", "name": "A", "allow_full_parcels": False},
            _h(KEY_B): {"client_id": "client-b", "name": "B", "allow_full_parcels": True},
        },
        _fh,
    )

os.environ["SAPIENT_ENGINE"] = "mock"
os.environ["SAPIENT_KEYS_FILE"] = KEYS_PATH
os.environ["SAPIENT_RATE_LIMIT_PER_MIN"] = "100000"
os.environ["SAPIENT_DAILY_QUOTA"] = "100000"
os.environ["SAPIENT_DOCS"] = "1"

from fastapi.testclient import TestClient  # noqa: E402
from sapient_serving.api.app import create_app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture
def key_a() -> str:
    return KEY_A


@pytest.fixture
def key_b() -> str:
    return KEY_B

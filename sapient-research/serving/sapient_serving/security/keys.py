"""API-key store.

We never store raw keys. A key is `sk_sapient_<random>`; we persist only its
SHA-256 hash mapped to a client record. Verifying a request hashes the presented
key and looks it up. If the file is leaked, the keys themselves are not exposed
(an attacker would need a SHA-256 preimage).

For a few clients on a single instance this file-backed store is fine. For
multi-instance / higher assurance, back this with a managed secret store or DB
and rotate keys per client.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass


KEY_PREFIX = "sk_sapient_"


@dataclass(frozen=True)
class Client:
    client_id: str
    name: str
    allow_full_parcels: bool = False


def generate_key() -> str:
    """Mint a new raw client key. Show this to the client ONCE; store only its hash."""
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.strip().encode("utf-8")).hexdigest()


class KeyStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self._by_hash: dict[str, Client] = {}

    def load(self) -> "KeyStore":
        self._by_hash = {}
        if not os.path.exists(self.path):
            return self  # default-deny: no keys file => nobody is authorized
        with open(self.path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for key_hash, rec in data.items():
            self._by_hash[key_hash] = Client(
                client_id=rec["client_id"],
                name=rec.get("name", rec["client_id"]),
                allow_full_parcels=bool(rec.get("allow_full_parcels", False)),
            )
        return self

    def verify(self, raw_key: str | None) -> Client | None:
        """Return the Client for a valid key, else None. Default-deny."""
        if not raw_key:
            return None
        presented = hash_key(raw_key)
        # Constant-time compare against each stored hash to avoid leaking which
        # prefix matched via timing. (Small N of clients => negligible cost.)
        for stored_hash, client in self._by_hash.items():
            if hmac.compare_digest(presented, stored_hash):
                return client
        return None

    def add(self, client_id: str, name: str, allow_full_parcels: bool = False) -> str:
        """Mint + persist a new key; returns the RAW key (caller must surface it once)."""
        raw = generate_key()
        data: dict = {}
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        data[hash_key(raw)] = {
            "client_id": client_id,
            "name": name,
            "allow_full_parcels": allow_full_parcels,
        }
        # 0600 perms: only the owner can read the (hashed) key store.
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        return raw

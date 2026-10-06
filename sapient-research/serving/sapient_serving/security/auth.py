"""Default-deny authentication dependency.

Every /v1 route depends on `current_client`. A request with no key, or an
unknown key, gets 401 and never reaches the model. The key may be presented as
`Authorization: Bearer <key>` or `X-API-Key: <key>`.
"""
from __future__ import annotations

from fastapi import Header, HTTPException, status

from .keys import Client, KeyStore

# Built once at startup (see api/deps.py) and reused.
_keystore: KeyStore | None = None


def init_keystore(path: str) -> KeyStore:
    global _keystore
    _keystore = KeyStore(path).load()
    return _keystore


def _extract_key(authorization: str | None, x_api_key: str | None) -> str | None:
    if x_api_key:
        return x_api_key.strip()
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


async def current_client(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> Client:
    if _keystore is None:
        # Misconfiguration => deny, don't fail open.
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="auth not initialized")
    client = _keystore.verify(_extract_key(authorization, x_api_key))
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return client

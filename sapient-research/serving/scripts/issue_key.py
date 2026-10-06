#!/usr/bin/env python3
"""Mint a client API key.

Prints the RAW key ONCE (hand it to the client over a secure channel). Only the
SHA-256 hash is written to the key store, so the key cannot be recovered from disk.

Examples:
    python scripts/issue_key.py --client-id acme
    python scripts/issue_key.py --client-id bigco --name "BigCo Labs" --allow-full-parcels
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sapient_serving.security.keys import KeyStore  # noqa: E402
from sapient_serving.settings import settings  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Issue a Sapient-1 API key.")
    ap.add_argument("--client-id", required=True, help="stable identifier for this client")
    ap.add_argument("--name", default=None, help="human-readable client name")
    ap.add_argument(
        "--allow-full-parcels",
        action="store_true",
        help="permit this client to request the full 1000-dim parcel vector (default: no)",
    )
    ap.add_argument("--keys-file", default=settings.keys_file)
    args = ap.parse_args()

    store = KeyStore(args.keys_file)
    raw = store.add(args.client_id, args.name or args.client_id, args.allow_full_parcels)

    print(f"client_id          : {args.client_id}")
    print(f"keys_file          : {args.keys_file}")
    print(f"allow_full_parcels : {args.allow_full_parcels}")
    print(f"API key            : {raw}")
    print("\nGive this key to the client now — it is not stored and cannot be recovered.")


if __name__ == "__main__":
    main()

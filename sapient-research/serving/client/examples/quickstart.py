"""Minimal end-to-end usage of the Sapient-1 client.

    pip install ./client requests
    python client/examples/quickstart.py

Set SAPIENT_API_KEY and (optionally) SAPIENT_API_URL first.
"""
from __future__ import annotations

import os

from sapient_client import Client

client = Client(
    api_key=os.environ["SAPIENT_API_KEY"],
    base_url=os.environ.get("SAPIENT_API_URL", "http://localhost:8000"),
)

print("provenance:", client.info()["served_model"])

# Transcript-only is fine; add video=/audio= to send media.
result = client.encode(transcript="A short clip of waves crashing on a beach at sunset.")
print("ROI scores:")
for roi, value in result["roi_scores"].items():
    print(f"  {roi:16s} {value:.3f}")
print("parcels:", result["parcels"]["n_parcels"], "| image bytes:", len(result["image_data_uri"]))

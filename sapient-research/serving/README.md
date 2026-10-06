# sapient-serving

A secure, access-controlled hosting layer for the **Sapient-1** brain encoder.
Clients send a stimulus (video / audio / transcript) and get back a coarse,
paper-faithful response — **per-ROI activation + a Schaefer-1000 parcel summary +
a brain image** — over an authenticated API. **The model weights never leave the
server.**

> **Provenance, stated plainly:** this serves **Sapient-1** — a clean-room,
> CC0-trained *trimodal* encoder (V-JEPA2 + W2V-BERT + Llama-3.2-3B; `d_model=1152`;
> 4 subjects). It is **NOT "Mary"**, the 6-stream research model in the papers, and
> it does not reproduce the papers' reported metrics. `GET /v1/info` says so.

## Two modes

| Mode | What runs | Needs |
|---|---|---|
| **mock** (default) | API + playground + client end-to-end with shaped, fake outputs | nothing heavy — runs today |
| **real** | the true Sapient-1 forward pass on GPU | the `[real]` extra, the private `sapient1` package, a checkpoint, Modal |

Mock mode exists so clients can experiment with the **live interface** before the
checkpoint is trained/uploaded. Flip `SAPIENT_ENGINE=real` once the model is ready.

## Quickstart (local, mock)

```bash
make setup                              # pip install -e ".[dev]"
python scripts/issue_key.py --client-id demo    # prints an API key (store it)
make dev                                # serve on http://localhost:8000
# open http://localhost:8000/playground , paste the key, upload a clip
```

curl:

```bash
KEY=sk_sapient_...      # from issue_key.py
curl -s -X POST localhost:8000/v1/encode -H "Authorization: Bearer $KEY" \
     -F "transcript=waves crashing at sunset" | tee /tmp/j.json
JOB=$(python -c "import json;print(json.load(open('/tmp/j.json'))['job_id'])")
curl -s localhost:8000/v1/jobs/$JOB -H "Authorization: Bearer $KEY"
```

Python client (the thin shim you hand to clients — no model code inside):

```bash
pip install ./client
SAPIENT_API_KEY=$KEY python client/examples/quickstart.py
```

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/healthz` | — | liveness |
| GET | `/v1/info` | key | model + provenance (honest) |
| POST | `/v1/encode` | key | submit stimulus → `{job_id}` |
| GET | `/v1/jobs/{id}` | key | poll status / fetch result (owner-only) |
| GET | `/playground` | — | static web UI (asks the user for their key) |

## Layout

```
sapient_serving/
  settings.py            env-driven config
  schemas.py             request/response contract
  security/              keys (hashed), auth (default-deny), ratelimit (per-client)
  engine/                base · mock · real · features · postprocess · plotting · provenance
  storage/jobs.py        ephemeral, tenant-scoped, TTL'd result store
  api/                   FastAPI app + routes
  modal_app.py           GPU deploy (real mode)
playground/index.html    the experiment UI
client/                  sapient-client (thin HTTP shim)
tests/                   auth · isolation · flow
scripts/issue_key.py     mint client keys
SECURITY.md              threat model + controls + dev→prod checklist
```

## Going from mock → real
1. Upload a real checkpoint to `SAPIENT_MODEL_ID` on HF (per the repo README it's not yet uploaded).
2. Install the `[real]` extra + the `sapient1` package on the GPU box.
3. Wire `engine/features.py` to the repo's `data/extract_*` and load the canonical
   Schaefer-1000→ROI mapping in `engine/real.py` (`_roi_network_ids`).
4. `make deploy` (Modal), then `SAPIENT_ENGINE=real`.

See **SECURITY.md** before exposing this to anyone.

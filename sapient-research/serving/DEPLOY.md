# Deploying the Sapient sandbox

Goal: a shareable URL + per-client keys you can hand to Robert / investors. The
sandbox runs the **mock** engine — secure, CPU-only, no weights, nothing to leak —
which is the right artifact for an external demo. (The real `digital_brain` engine
is opt-in; see the last section.)

## 0. Issue a key (once)
```bash
python scripts/issue_key.py --client-id investor-demo   # prints sk_sapient_... ONCE
# this writes keys.json (only the hash is stored)
```
Keep `keys.json`; you'll provide it to the host as a secret file. Never commit it.

## 1. Run with Docker (local or any VM)
```bash
docker build -t sapient-sandbox .
docker run -p 8000:8000 -v "$PWD/keys.json:/data/keys.json:ro" sapient-sandbox
# open http://localhost:8000/demo , paste the key
```

## 2. Render (easiest shareable URL)
1. Push this folder to a Git repo and connect it on https://render.com (uses `render.yaml`).
2. In the service's **Secret Files**, add `keys.json` (the file from step 0) mounted at
   `/etc/secrets/keys.json` (already referenced in `render.yaml`).
3. Deploy → you get `https://sapient-sandbox.onrender.com`. Share that + each client's key.

## 3. Fly.io (alternative)
```bash
fly launch --no-deploy            # generates fly.toml from the Dockerfile
fly secrets set ...               # or mount keys.json
fly deploy
```
Set `internal_port = 8000` and a `[[http_service]]` health check on `/healthz`.

## 4. Modal (if you already use it)
Deploy the FastAPI app as a CPU asgi_app (no GPU for the sandbox). See `modal_app.py`
for the GPU/real-engine variant; for the sandbox, force `SAPIENT_ENGINE=mock` and drop the `gpu=` arg.

## Per-client access
Issue one key per recipient so you can revoke/rotate individually and attribute usage:
```bash
python scripts/issue_key.py --client-id robert
python scripts/issue_key.py --client-id acme-ventures
```
All keys live (hashed) in the one `keys.json` you give the host.

## Optional: the REAL Digital Brain engine (`digital_brain`)
This serves the published Digital Brain *visual-cortex* model (real predictions;
faces/scenes/bodies/text/vision). Heavier (CLIP ViT-L/14 ~1.7 GB, CPU) and image-based.
On a box that has the model + repo:
```bash
pip install ".[real]" transformers pillow
export SAPIENT_ENGINE=digital_brain
export SAPIENT_DB_REPO=/path/to/digital-brain          # for unpickling src.geometry_aware_encoder
export SAPIENT_DB_MODEL=/path/to/geo_subj05_N4.pkl
# pin scikit-learn==1.6.1 to match the version the model was trained with
```
`/v1/info` will state honestly that purchase-intent in this mode is a *visual-engagement
proxy* (visual cortex), not a reward measurement.

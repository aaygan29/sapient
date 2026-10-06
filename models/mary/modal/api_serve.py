"""mary-serve-api — DEDICATED warm Mary brain-encoder serving for API v2 traffic.

SAFETY / ISOLATION: This is its OWN deployed Modal app (`mary-serve-api`), with
its OWN warm GPU container pool, completely independent of the LIVE, UI-facing
`mary-serve` app (mary/modal/serve.py). API traffic hitting this app NEVER
competes with the live product's containers for GPU, and a deploy here can never
disturb the live `submit` endpoint the product depends on. (Mirrors how
`mary-whisper` / `mary-crossmodal-serve` are separate apps reusing the same
shared secret + image.)

REUSE, NOT FORK: every line of serving logic — the warm `load()`, the Yeo-7
network reducers, the per-second KPI + networkTimeSeries block, the storage
helpers, `_build_artifact`, and `run_job` — is INHERITED unchanged from
`serve._MaryServerBase`. The SAME container `image`, `VOLUMES`, and `SECRETS` are
imported from `serve` too. The only thing redefined here is the `@app.cls(...)`
deploy config (its own pool) and a thin `submit` web endpoint. This guarantees
the API and UI run byte-identical model code while staying physically isolated.

WS-A surface (already in _MaryServerBase, served by this app):
  • artifact["networkTimeSeries"] — {<Yeo-7 name>: [per_second_float, …]} for the
    7 Yeo networks, bucketed on the SAME per-second edges as kpiTimeSeries (so
    each series length == kpiTimeSeries length), rounded to 6 dp.
  • include_fmri (request flag) — when true, the raw (T_TR, 20484) predicted
    verts are persisted to Supabase storage kairo-uploads/fmri/<run_id>.npz
    (np.savez_compressed) and artifact["fmri_path"] is set. Default false → no-op
    (keeps payloads small).

Pool config: gpu="A100-40GB", min_containers=1 (always warm), max_containers=4
(independent of the live app's 1–2). The web `submit()` shape matches the live
app exactly (auth_token + run_id + capability/modality/input_*/analysis_mode),
plus the optional `include_fmri` flag.

Deploy (ONLY this app — never `modal deploy serve.py` from here):
  modal deploy mary/modal/api_serve.py
Smoke:
  modal run mary/modal/api_serve.py::smoke
"""
from __future__ import annotations

import os

import modal

# Reuse the EXACT image / volumes / secrets / serving logic from the live module.
# Importing `serve` also registers the live `mary-serve` app object, but we never
# reference or deploy it here — `modal deploy api_serve.py` only publishes the
# app defined in THIS file.
from serve import (
    SECRETS, VOLUMES, _MaryServerBase, _SlowFastExtractorBase, image, slowfast_image,
)

# The image inherited from `serve` bundles the project dirs (via add_local_dir)
# but NOT serve.py itself — yet this app's class + endpoint `import serve` at
# CONTAINER runtime (MaryServerAPI inherits serve._MaryServerBase). Without this
# the container crashes on boot with "ModuleNotFoundError: No module named
# 'serve'" and every request hangs until timeout. Bundle serve.py into the image.
image = image.add_local_python_source("serve")
# The DEDICATED SlowFast extractor for this app (SlowFastExtractorAPI below) runs
# in `slowfast_image`. Its container imports THIS module at boot, which does
# `from serve import ...` at top level — so both `serve` and `api_serve` must be
# importable inside slowfast_image, or the SlowFast container crash-loops on
# ModuleNotFoundError and the visual stream is (again) silently unavailable.
slowfast_image = slowfast_image.add_local_python_source("serve", "api_serve")

API_APP_NAME = "mary-serve-api"

app = modal.App(API_APP_NAME)


# DEDICATED visual extractor for THIS app. A Modal `@app.cls` can only be
# `.remote()`-called from within the app it's registered on. `serve.SlowFastExtractor`
# is bound to the live `mary-serve` app, so calling it from `mary-serve-api` raised
# `ExecutionError: Function has not been hydrated ... the App it is defined on is not
# running` — and EVERY video job on the API pool silently fell back to audio-only
# (the bug that collapsed two different IG reels to r=0.99). Re-registering the SAME
# extractor logic on this app makes the visual/SlowFast stream actually run for API
# traffic. (We reuse serve.SlowFastExtractor's body via subclassing so the featurize
# math stays byte-identical to the training contract.)
@app.cls(
    image=slowfast_image, gpu="A10G", volumes=VOLUMES,
    secrets=[modal.Secret.from_name("hf-token")],
    min_containers=0, scaledown_window=300, timeout=10 * 60,
)
class SlowFastExtractorAPI(_SlowFastExtractorBase):
    """SlowFast R101 visual featurizer registered on the `mary-serve-api` app.
    Identical to the live `serve.SlowFastExtractor` (load + featurize inherited
    from `_SlowFastExtractorBase`); only the @app.cls binding differs so API-pool
    jobs can `.remote()`-call it."""
    pass


@app.cls(
    image=image, gpu="A100-40GB", volumes=VOLUMES, secrets=SECRETS,
    # DEDICATED API POOL — independent of the live UI app's 1–2 containers.
    # min_containers=1 keeps one A100 always warm for low-latency API calls;
    # max_containers=4 lets concurrent API jobs fan out across up to four GPUs
    # without ever borrowing from (or being throttled by) the live product pool.
    min_containers=1, max_containers=4, scaledown_window=300, timeout=20 * 60,
)
class MaryServerAPI(_MaryServerBase):
    """API-traffic serving class. All logic inherited from `_MaryServerBase`
    (load + reducers + run_job + the WS-A networkTimeSeries / include_fmri
    additions). Only the deploy config above differs from the live class."""

    def _slowfast_extractor_cls(self):
        # Use THIS app's extractor (see SlowFastExtractorAPI above) — not the one
        # bound to the live `mary-serve` app, which is unreachable from here.
        return SlowFastExtractorAPI


@app.function(image=image, secrets=SECRETS, timeout=60)
@modal.fastapi_endpoint(method="POST")
def submit(payload: dict) -> dict:
    """API v2 → Modal trigger for the DEDICATED pool. Same shape + auth as the
    live `mary-serve` submit, plus the optional `include_fmri` flag. Verifies the
    shared secret, spawns the GPU job on THIS app's warm pool, and returns the
    modal call id immediately (the job writes status/result back to mary_runs)."""
    from fastapi import HTTPException

    expected = os.environ.get("MARY_PIPELINE_SECRET")
    if not expected or payload.get("auth_token") != expected:
        raise HTTPException(status_code=401, detail="bad auth_token")
    run_id = payload.get("run_id")
    capability = payload.get("capability", "brain_map")
    modality = payload.get("modality", "text")
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id required")
    call = MaryServerAPI().run_job.spawn(
        run_id=run_id, capability=capability, modality=modality,
        input_text=payload.get("input_text"), input_url=payload.get("input_url"),
        analysis_mode=payload.get("analysis_mode", "full"),
        include_fmri=bool(payload.get("include_fmri", False)),
    )
    return {"modal_call_id": call.object_id}


@app.local_entrypoint()
def smoke(text: str = "The waves crashed against the rocks as the sun set over the quiet harbor."):
    """End-to-end text smoke against the dedicated API pool — exercises the same
    probe path the live smoke uses, plus prints the WS-A networkTimeSeries shape
    so you can confirm 7 keys each length == kpiTimeSeries length."""
    res = MaryServerAPI().probe.remote(text=text)
    print("smoke score:", res.get("score"), "duration_sec:", res.get("duration_sec"))
    nts = res.get("networkTimeSeries") or {}
    kts = res.get("kpiTimeSeries") or {}
    klen = len(next(iter(kts.values()), [])) if kts else 0
    print(f"networkTimeSeries: {len(nts)} networks; "
          f"lengths={{ {', '.join(f'{k}:{len(v)}' for k, v in nts.items())} }}")
    print(f"kpiTimeSeries length: {klen}  (each networkTimeSeries series should match)")
    print("headline:", res.get("headline"))

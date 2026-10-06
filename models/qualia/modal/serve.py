"""qualia/modal/serve.py — Qualia (Kairo v4) SERVING app (Modal). Formerly kairo/kairo_serve.py.

Stands Kairo up behind a warm Modal class so the API can call it. Given a Friends
clip id (features are cached on `kairo-features`), it:
  1. loads the 18 streams in the EXACT trained order (ckpt['streams']),
  2. runs Kairo (crossstream_mix wiring — the one proven to differentiate),
  3. produces native 1000 Schaefer parcels per second,
  4. reduces to the 7 Yeo networks via the shipped Schaefer-1000→Yeo7 CSV,
  5. returns Mary's artifact shape: networkTimeSeries = {<Yeo7 name>: [per_sec...]}.

Entrypoints:
  • Kairo.score (method)      — clip_id -> artifact dict (the served call).
  • score_clip (function)     — thin wrapper for the local entrypoint / API.
  • gate (function)           — runs TWO DIFFERENT real videos (Friends clips) through
                                the served path and reports whether outputs DIFFER.

ARBITRARY URLs: Kairo needs 8 backbones / 18 streams (vjepa, videomae2,
internvl3-8b-8bit, emonet, whisper, llama-3.2-3b, w2v-bert, vjepa2). The
`kairo-extract` app rebuilds these from a raw video URL. As of the VISUAL-ON
fix, the full sensory front-end is live: whisper + w2v-bert + llama (audio/text),
PLUS V-JEPA ViT-L (vjepa_block{5,15,23}), VideoMAEv2-huge (vmae2_block{10,18,25}),
V-JEPA2 ViT-g (vjepa2) and InternVL3-8B (ivl3_*). Only `emonet` (a custom face
net not on HF) remains zero-filled. So end-to-end raw-URL scoring with the full
visual stack is supported (gate_url / run_job / submit).
"""
from __future__ import annotations
import modal

# PUBLIC NAME: this serving app is "Qualia". The Modal app is deployed as
# `qualia-serve` (overridable via QUALIA_APP_NAME for a phased rename). The
# legacy `kairo-serve` deploy can remain live as a fallback while the API is
# pointed at the new name; the internal model/extract code keeps the `kairo`
# ids on purpose (the Kairo v4 checkpoint + the `kairo-extract`/`kairo-weights`/
# `kairo-features` volumes are named that way and renaming them risks the live
# extraction path).
import os as _os
import sys as _sys

# This serving app lives in `qualia/modal/serve.py`; the encoder module it loads
# (`model.py`, formerly `kairo_model.py`) lives in the sibling `qualia/serving/` dir.
# Put that dir on sys.path so BOTH `add_local_python_source("model")` (deploy-time)
# and `from model import ...` (in-container) resolve regardless of CWD.
_SERVING_DIR = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "serving")
)
if _SERVING_DIR not in _sys.path:
    _sys.path.insert(0, _SERVING_DIR)

APP_NAME = _os.environ.get("QUALIA_APP_NAME", "qualia-serve")
app = modal.App(APP_NAME)

# The submit endpoint + the async write-back job need: supabase (to finalize the
# mary_runs row, identically to mary-serve) and the shared pipeline secret (to
# auth the Express → Modal trigger). Reuse Mary's existing secrets so no new
# secret has to be created.
SECRETS = [
    modal.Secret.from_name("mary-supabase"),       # SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY
    modal.Secret.from_name("mary-pipeline-secret"), # MARY_PIPELINE_SECRET (== KAIRO_PIPELINE_SECRET)
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy>=1.26,<3", "h5py>=3.10",
                 "supabase>=2", "fastapi[standard]")
    .add_local_python_source("model")
)

weights_vol = modal.Volume.from_name("kairo-weights")
features_vol = modal.Volume.from_name("kairo-features")
extracted_vol = modal.Volume.from_name("kairo-extracted", create_if_missing=True)
VOLUMES = {"/weights": weights_vol, "/features": features_vol, "/extracted": extracted_vol}
CKPT = "/weights/v4_seed_0/best_model.pt"

# STREAM_DIMS — keep in lockstep with kairo_extract.STREAM_DIMS / kairo_model.FEATURE_DIMS.
STREAM_DIMS = {
    "vjepa_block5": 1024, "vjepa_block15": 1024, "vjepa_block23": 1024,
    "whisper_layer12": 1280, "whisper_layer25": 1280, "whisper_layer31": 1280,
    "whisper_layernorm": 1280, "vmae2_block10": 1280, "vmae2_block18": 1280,
    "vmae2_block25": 1280, "ivl3_layer10": 3584, "ivl3_layer15": 3584,
    "ivl3_layer20": 3584, "ivl3_norm": 3584, "emonet": 900, "llama": 9216,
    "w2vbert": 3072, "vjepa2": 4224,
}

# Canonical Yeo-7 names — MUST match Mary's YEO7_ORDER (src/lib/maryBrain.ts union).
# CSV labels are 1..7 in Schaefer's standard Yeo7 order:
#   1 Vis, 2 SomMot, 3 DorsAttn, 4 SalVentAttn, 5 Limbic, 6 Cont, 7 Default
YEO7_ORDER = ["Visual", "Somatomotor", "Dorsal Attention",
              "Ventral Attention", "Limbic", "Frontoparietal", "Default Mode"]
YEO7_CSV = "/features/parcel_network_divisions/Schaefer_1000Parcel_7Networks_order_Yeo7.csv"


def _clip_to_episode(clip):
    return clip[:-1] if clip and clip[-1].isalpha() else clip


def _load_clip_feats(clip):
    """Load the 18 streams for a clip from cached .h5, strictly per ckpt['streams']."""
    import glob, os, h5py, numpy as np, torch
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    streams = ck["streams"]
    out = {}
    for s, (subdir, layer, dim) in streams.items():
        base = f"/features/{subdir}"
        cands = glob.glob(os.path.join(base, "**", f"friends_{clip}.h5"), recursive=True)
        if not cands:
            cands = glob.glob(os.path.join(base, "**", f"*{clip}*.h5"), recursive=True)
        if not cands:
            raise FileNotFoundError(f"{s}: no h5 for {clip} under {base}")
        path = sorted(cands)[0]
        with h5py.File(path, "r") as f:
            if layer in f:
                arr = f[layer][:]
            else:
                ep = _clip_to_episode(clip)
                grp = None
                for gk in (ep, clip):
                    if gk in f:
                        grp = f[gk]; break
                if grp is None:
                    grp = f[list(f.keys())[0]]
                lk = layer.split("/")[-1]
                arr = grp[lk][:] if lk in grp else grp[list(grp.keys())[0]][:]
        arr = np.asarray(arr, dtype=np.float32)
        while arr.ndim > 2:
            arr = arr.squeeze(1) if arr.shape[1] == 1 else arr.reshape(arr.shape[0], -1)
        if arr.shape[-1] != dim:
            arr = arr[:, :dim] if arr.shape[-1] > dim else np.pad(arr, ((0,0),(0,dim-arr.shape[-1])))
        out[s] = arr
    T = min(a.shape[0] for a in out.values())
    return {s: a[:T] for s, a in out.items()}


@app.cls(image=image, volumes=VOLUMES, cpu=8, memory=32768, timeout=1800,
         min_containers=0, scaledown_window=300)
class Kairo:
    @modal.enter()
    def load(self):
        import numpy as np, torch
        from model import load_kairo
        self.torch = torch
        self.np = np
        self.model, self.meta, report = load_kairo(CKPT)
        self.model.eval()
        self.clean = not report["missing"] and not report["unexpected"]
        self.labels = np.loadtxt(YEO7_CSV, dtype=int)   # (1000,) network id 1..7
        print(f"[kairo-serve] loaded v4 clean={self.clean} val_pearson={self.meta.get('val_pearson')}")

    def _forward(self, feats, subject):
        torch = self.torch
        from model import STREAM_ORDER
        t = {s: torch.from_numpy(feats[s])[None] for s in STREAM_ORDER}
        subj = torch.tensor([subject], dtype=torch.long)
        projected = [self.model.encoder.projections[s](t[s]) for s in STREAM_ORDER]
        stk = torch.stack(projected, dim=2)                 # (1,T,18,256)
        B, T, S, C = stk.shape
        mixed = self.model.encoder.transformer(stk.reshape(B*T, S, C)).reshape(B, T, S, C)
        subj_e = self.model.encoder.subject_emb(subj)       # (1,256)
        subj_t = subj_e[:, None, None, :].expand(B, T, 1, C)
        cat = torch.cat([mixed, subj_t], dim=2).reshape(B, T, (S+1)*C)
        with torch.no_grad():
            h = self.model.bottleneck(cat)
            parcels = self.model.predictor(h)               # (1,T,1000)
        return parcels.squeeze(0).numpy()

    @modal.method()
    def score(self, clip_id: str, subject: int = 0):
        """clip_id -> Mary-shaped artifact with per-second Yeo-7 networkTimeSeries."""
        np = self.np
        feats = _load_clip_feats(clip_id)
        parcels = self._forward(feats, subject)             # (T,1000) signed
        n_sec = parcels.shape[0]
        # per-second Yeo-7 network means (signed parcel activations)
        net_per_sec = np.zeros((n_sec, 7), dtype=np.float64)
        for i in range(7):
            m = self.labels == (i + 1)
            if m.any():
                net_per_sec[:, i] = parcels[:, m].mean(1)
        network_ts = {
            YEO7_ORDER[i]: [round(float(net_per_sec[s, i]), 6) for s in range(n_sec)]
            for i in range(7)
        }
        return {
            "model": "kairo-v4",
            "clip_id": clip_id,
            "subject": subject,
            "n_seconds": int(n_sec),
            "networkTimeSeries": network_ts,                # Mary artifact shape
            "meta": {"val_pearson": float(self.meta.get("val_pearson", 0.0)),
                     "load_clean": bool(self.clean), "n_parcels": 1000},
        }

    @modal.method()
    def score_npz(self, npz_path: str, subject: int = 0, npz_bytes: bytes = None):
        """extract_all features -> Mary-shaped artifact. `npz_bytes` (the in-memory
        npz returned by extract_all) is preferred and fully race-free; `npz_path` is
        the shared-volume fallback."""
        np = self.np
        feats = _load_url_feats(npz_bytes if npz_bytes is not None else npz_path)
        parcels = self._forward(feats, subject)             # (T,1000) signed
        n_sec = parcels.shape[0]
        net_per_sec = np.zeros((n_sec, 7), dtype=np.float64)
        for i in range(7):
            m = self.labels == (i + 1)
            if m.any():
                net_per_sec[:, i] = parcels[:, m].mean(1)
        network_ts = {
            YEO7_ORDER[i]: [round(float(net_per_sec[s, i]), 6) for s in range(n_sec)]
            for i in range(7)
        }
        return {
            "model": "kairo-v4",
            "npz": npz_path,
            "subject": subject,
            "n_seconds": int(n_sec),
            "networkTimeSeries": network_ts,
            "meta": {"val_pearson": float(self.meta.get("val_pearson", 0.0)),
                     "load_clean": bool(self.clean), "n_parcels": 1000},
        }


def _load_url_feats(npz_src):
    """Load the 18 streams (in STREAM_ORDER) from extract_all output, zero-filling
    any stream absent at the correct trained dim, trimmed to common T. `npz_src` is
    either the raw npz BYTES that extract_all returned in-memory (preferred, fully
    race-free) or a path string to the npz on the shared /extracted volume."""
    import numpy as np
    import io, time
    if isinstance(npz_src, (bytes, bytearray)):
        # Race-free path: features came back in the function return, no volume read.
        z = np.load(io.BytesIO(npz_src))
    else:
        # Fallback: read the file off the shared volume. A warm scoring container
        # holds a stale snapshot, so reload + retry to cover commit propagation.
        z = None
        last_err = None
        for _ in range(8):
            try:
                extracted_vol.reload()
            except Exception:
                pass
            try:
                z = np.load(npz_src)
                break
            except FileNotFoundError as e:
                last_err = e
                time.sleep(1.5)
        if z is None:
            raise last_err
    keys = set(z.files)
    T = 0
    for s in STREAM_DIMS:
        if s in keys and z[s].shape[0] > 0:
            T = z[s].shape[0]; break
    out = {}
    for s, dim in STREAM_DIMS.items():
        if s in keys and z[s].shape[0] >= T and T > 0:
            a = np.asarray(z[s][:T], dtype=np.float32)
            # scrub NaN/inf (e.g. InternVL3 final-norm under 8-bit) so one bad value
            # can't turn every parcel into NaN. Defensive: also covers cached npzs.
            out[s] = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            out[s] = np.zeros((T, dim), dtype=np.float32)
    return out


@app.function(image=image, volumes=VOLUMES, timeout=1800)
def score_clip(clip_id: str = "s01e01a", subject: int = 0):
    return Kairo().score.remote(clip_id, subject)


@app.function(image=image, volumes=VOLUMES, timeout=5400)
def score_url(url: str, subject: int = 0, with_internvl: bool = False):
    """END-TO-END: arbitrary video URL → all 18 streams (kairo-extract) → Kairo →
    Mary-shaped artifact with per-second Yeo-7 networkTimeSeries. The missing
    backbones are rebuilt in the `kairo-extract` app; any unmatched stream is
    zero-filled (reported in `extract_manifest`)."""
    extract_all = modal.Function.from_name("kairo-extract", "extract_all")
    manifest = extract_all.remote(url, with_internvl)
    art = Kairo().score_npz.remote(manifest["npz"], subject, manifest.get("npz_bytes"))
    art["source_url"] = url
    art["transcript"] = manifest.get("transcript")  # spoken transcript (or None)
    art["extract_manifest"] = {"real": manifest["real"], "zero_filled": manifest["zero_filled"],
                               "T": manifest["T"], "npz": manifest["npz"]}
    return art


@app.function(image=image, volumes=VOLUMES, timeout=2400)
def gate(clip_a: str = "s01e01a", clip_b: str = "s01e15a", subject: int = 0):
    """Run TWO DIFFERENT real videos through the SERVED path; report if they DIFFER."""
    import numpy as np
    k = Kairo()
    ra = k.score.remote(clip_a, subject)
    rb = k.score.remote(clip_b, subject)
    ra2 = k.score.remote(clip_a, subject)   # determinism control

    def stack(r):
        return np.array([r["networkTimeSeries"][n] for n in YEO7_ORDER])  # (7, T)

    A, B, A2 = stack(ra), stack(rb), stack(ra2)
    T = min(A.shape[1], B.shape[1])

    def net_corr(X, Y):
        out = []
        for i in range(7):
            x, y = X[i, :T], Y[i, :T]
            if x.std() > 1e-9 and y.std() > 1e-9:
                out.append(float(np.corrcoef(x, y)[0, 1]))
            else:
                out.append(float("nan"))
        return out

    same = net_corr(A, A2)
    diff = net_corr(A, B)
    print("\n=========== KAIRO-SERVE 2-VIDEO GATE (served path) ===========")
    print(f"A={clip_a} (n_sec={ra['n_seconds']})   B={clip_b} (n_sec={rb['n_seconds']})")
    print("  network            SAME(A,A)     DIFF(A,B)")
    for i, n in enumerate(YEO7_ORDER):
        print(f"  {n:<18}{same[i]:>9.3f}{diff[i]:>13.3f}")
    same_m = float(np.nanmean(same)); diff_m = float(np.nanmean(diff))
    differentiates = (same_m > 0.999) and (diff_m < 0.6)
    print(f"\n  mean SAME={same_m:.4f}  mean DIFF={diff_m:.4f}  margin={same_m-diff_m:.4f}")
    print(f"  >>> SERVED KAIRO DIFFERENTIATES THE TWO VIDEOS: {'YES' if differentiates else 'NO'} <<<")
    return {"clip_a": clip_a, "clip_b": clip_b, "same_mean": same_m, "diff_mean": diff_m,
            "same": same, "diff": diff, "differentiates": bool(differentiates),
            "n_sec_a": ra["n_seconds"], "n_sec_b": rb["n_seconds"]}


@app.function(image=image, volumes=VOLUMES, timeout=7200)
def gate_url(url_a: str, url_b: str, subject: int = 0, with_internvl: bool = False,
             skip_visual: bool = False):
    """THE ARBITRARY-URL PROOF: extract+score TWO different real video URLs through
    the full served path; report per-second differentiation. Same-URL-twice must be
    ~1.0; different URLs must be clearly lower."""
    import numpy as np
    extract_all = modal.Function.from_name("kairo-extract", "extract_all")
    ma = extract_all.remote(url_a, with_internvl, "gate_a", skip_visual)
    mb = extract_all.remote(url_b, with_internvl, "gate_b", skip_visual)
    k = Kairo()
    ra = k.score_npz.remote(ma["npz"], subject)
    rb = k.score_npz.remote(mb["npz"], subject)
    ra2 = k.score_npz.remote(ma["npz"], subject)   # determinism control

    def stack(r):  # (7, T)
        return np.array([r["networkTimeSeries"][n] for n in YEO7_ORDER])

    A, B, A2 = stack(ra), stack(rb), stack(ra2)
    T = min(A.shape[1], B.shape[1], A2.shape[1])

    def persec(X, Y):  # mean over seconds of the 7-vector correlation per second
        cs = []
        for t in range(T):
            x, y = X[:, t], Y[:, t]
            if x.std() > 1e-9 and y.std() > 1e-9:
                cs.append(float(np.corrcoef(x, y)[0, 1]))
        return float(np.mean(cs)) if cs else float("nan")

    def netcorr(X, Y):  # per-network timecourse correlation
        out = []
        for i in range(7):
            x, y = X[i, :T], Y[i, :T]
            out.append(float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-9 and y.std() > 1e-9 else float("nan"))
        return out

    same_ps, diff_ps = persec(A, A2), persec(A, B)
    same_net, diff_net = netcorr(A, A2), netcorr(A, B)
    print("\n========= KAIRO-SERVE ARBITRARY-URL 2-VIDEO GATE =========")
    print(f"A={url_a}\n  real={ma['real']}\n  zero={ma['zero_filled']} T={ma['T']}")
    print(f"B={url_b}\n  real={mb['real']}\n  zero={mb['zero_filled']} T={mb['T']}")
    print(f"\n  per-second  SAME(A,A)={same_ps:.4f}   DIFF(A,B)={diff_ps:.4f}")
    print("  network            SAME(A,A)     DIFF(A,B)")
    for i, n in enumerate(YEO7_ORDER):
        print(f"  {n:<18}{same_net[i]:>9.3f}{diff_net[i]:>13.3f}")
    differentiates = (same_ps > 0.999) and (diff_ps < 0.9)
    print(f"\n  >>> DIFFERENTIATES TWO REAL URLS: {'YES' if differentiates else 'NO'} <<<")
    return {"url_a": url_a, "url_b": url_b, "same_persec": same_ps, "diff_persec": diff_ps,
            "same_net": same_net, "diff_net": diff_net, "differentiates": bool(differentiates),
            "manifest_a": {k2: ma[k2] for k2 in ("real", "zero_filled", "T")},
            "manifest_b": {k2: mb[k2] for k2 in ("real", "zero_filled", "T")},
            "n_sec_a": ra["n_seconds"], "n_sec_b": rb["n_seconds"]}


# ─────────────────────────────────────────────────────────────────────────────
# ASYNC SERVING — mirrors mary/modal/serve.py's submit→spawn→write-back pattern.
#
# WHY: the Express API handler is serverless (≤60s). score_url downloads +
# extracts + scores a video, which takes minutes → the old SYNCHRONOUS in-handler
# call (api/_qualiaRun.ts) blew the function timeout → 502. The fix is the SAME
# async pattern Mary uses: a `submit` web endpoint spawns a GPU/CPU job that
# writes status + the finished artifact back into the `mary_runs` Supabase table,
# and the frontend polls. We reuse Mary's mary_runs columns + artifact SHAPE so
# the consumer UI (MaryArtifactPanel) + the public API shaper (_scanShape) render
# a Qualia run with ZERO changes.
#
# The artifact builder below is the Python port of api/_qualiaRun.ts
# buildQualiaArtifact (kept byte-for-byte equivalent in math), so a run finalized
# here looks identical to one the old sync path produced.
# ─────────────────────────────────────────────────────────────────────────────

YEO7_PLAIN = {
    "Visual": "Seeing & imagery",
    "Somatomotor": "Body & movement",
    "Dorsal Attention": "Focused attention",
    "Ventral Attention": "Salience & surprise",
    "Limbic": "Emotion & reward",
    "Frontoparietal": "Reasoning & effort",
    "Default Mode": "Reflection & self",
}
KPI_NAMES = [
    "Visual Attention", "Auditory Engagement", "Face Processing", "Reading Engagement",
    "Language Comprehension", "Cognitive Effort", "Reward Valuation (cortical proxy)",
    "Emotional Salience", "Narrative Absorption", "Surprise / Novelty",
]
KPI_FROM_NETWORKS = {
    "Visual Attention": {"Visual": 0.65, "Dorsal Attention": 0.35},
    "Auditory Engagement": {"Somatomotor": 1.0},
    "Face Processing": {"Visual": 1.0},
    "Reading Engagement": {"Visual": 0.6, "Default Mode": 0.4},
    "Language Comprehension": {"Default Mode": 0.5, "Frontoparietal": 0.5},
    "Cognitive Effort": {"Frontoparietal": 1.0},
    "Reward Valuation (cortical proxy)": {"Limbic": 1.0},
    "Emotional Salience": {"Limbic": 0.6, "Ventral Attention": 0.4},
    "Narrative Absorption": {"Default Mode": 1.0},
    "Surprise / Novelty": {"Ventral Attention": 1.0},
}
COMPOSITE_FROM_KPIS = {
    "Voice Impact": {"Auditory Engagement": 0.6, "Language Comprehension": 0.4},
    "Visual Pull": {"Visual Attention": 0.6, "Face Processing": 0.25, "Reading Engagement": 0.15},
    "Cognitive Grip": {"Cognitive Effort": 0.5, "Language Comprehension": 0.3, "Surprise / Novelty": 0.2},
    "Emotional Hit": {"Emotional Salience": 0.6, "Reward Valuation (cortical proxy)": 0.4},
    "Memorability": {"Narrative Absorption": 0.6, "Reward Valuation (cortical proxy)": 0.2, "Surprise / Novelty": 0.2},
}
KPI_Z_SPREAD = 14.0
KPI_SCORE_LO = 12
KPI_SCORE_HI = 94


def build_qualia_artifact(art: dict, media_url: str | None) -> dict:
    """Port of api/_qualiaRun.ts buildQualiaArtifact — kairo-serve's per-second
    networkTimeSeries → the FULL Mary artifact (networks/score/kpiTimeSeries/
    kpiSummary/composites/verdictSummary/...). RELATIVE-within-run scoring (z vs
    the clip's own KPI mean), since kairo emits signed unit-scale activations."""
    import numpy as np
    nts = art.get("networkTimeSeries") or {}
    n_sec = max([len(nts.get(n) or []) for n in YEO7_ORDER] + [0])

    def r6(x):  # round to 6 dp
        return round(float(x), 6)

    net_idx = {n: i for i, n in enumerate(YEO7_ORDER)}
    net_per_sec = np.zeros((n_sec, 7), dtype=np.float64)
    for n, i in net_idx.items():
        seq = nts.get(n) or []
        for s in range(min(n_sec, len(seq))):
            try:
                net_per_sec[s, i] = float(seq[s])
            except (TypeError, ValueError):
                net_per_sec[s, i] = 0.0

    # KPI per-second = sum-normalized weighted blend of networks per second.
    kpi_ts = {}
    for kpi in KPI_NAMES:
        weights = KPI_FROM_NETWORKS[kpi]
        wsum = sum(weights.values()) or 1.0
        acc = np.zeros(n_sec, dtype=np.float64)
        for net, w in weights.items():
            j = net_idx.get(net)
            if j is None:
                continue
            acc += (w / wsum) * net_per_sec[:, j]
        kpi_ts[kpi] = acc

    kpi_summary = {}
    for kpi in KPI_NAMES:
        arr = kpi_ts[kpi]
        m = float(arr.mean()) if arr.size else 0.0
        sd = float(arr.std()) if arr.size else 0.0
        argmin = int(arr.argmin()) if arr.size else 0
        argmax = int(arr.argmax()) if arr.size else 0
        kpi_summary[kpi] = {
            "mean": r6(m), "std": r6(sd),
            "min": r6(arr[argmin]) if arr.size else 0.0,
            "max": r6(arr[argmax]) if arr.size else 0.0,
            "argmin_sec": argmin, "argmax_sec": argmax,
        }

    # RELATIVE-within-run KPI score (z vs the per-clip KPI-mean level).
    kpi_means = np.array([kpi_summary[k]["mean"] for k in KPI_NAMES], dtype=np.float64)
    clip_mean = float(kpi_means.mean()) if kpi_means.size else 0.0
    clip_std = float(kpi_means.std()) or 1e-9
    kpi_scores = {}
    for k in KPI_NAMES:
        z = (kpi_summary[k]["mean"] - clip_mean) / clip_std
        kpi_scores[k] = int(max(KPI_SCORE_LO, min(KPI_SCORE_HI, round(50 + z * KPI_Z_SPREAD))))

    composites = []
    for cname, weights in COMPOSITE_FROM_KPIS.items():
        wsum = sum(weights.values()) or 1.0
        cscore = int(round(sum((w / wsum) * kpi_scores.get(k, 0) for k, w in weights.items())))
        tier = "strength" if cscore >= 70 else "okay" if cscore >= 40 else "weakness"
        composites.append({"name": cname, "score": cscore, "tier": tier})

    overall = int(round(sum(c["score"] for c in composites) / len(composites))) if composites else 50
    grade = ("A" if overall >= 85 else "B" if overall >= 70 else "C" if overall >= 55
             else "D" if overall >= 40 else "F")

    # peak_multiplier: best per-second composite signal / its mean.
    comp_per_sec = np.zeros(n_sec, dtype=np.float64)
    for cname, weights in COMPOSITE_FROM_KPIS.items():
        wsum = sum(weights.values()) or 1.0
        for k, w in weights.items():
            comp_per_sec += (w / wsum) * kpi_ts.get(k, np.zeros(n_sec))
    cmean = float(comp_per_sec.mean()) or 1e-9
    cmax = float(comp_per_sec.max()) if comp_per_sec.size else 0.0
    peak_mult = round((cmax / cmean), 2) if cmean != 0 else 1

    networks = [
        {"network": name, "value": r6(net_per_sec[:, i].mean() if n_sec else 0.0),
         "plain": YEO7_PLAIN.get(name, name)}
        for i, name in enumerate(YEO7_ORDER)
    ]
    sorted_nets = sorted(networks, key=lambda n: n["value"], reverse=True)
    top = sorted_nets[:3]
    hi = max([n["value"] for n in networks] + [0.0]) or 1.0
    top_regions = [{"label": t["plain"], "value": r6(t["value"] / hi)} for t in top]
    headline = (f"Strongest response in {top[0]['plain'].lower()}" if top else "Brain response map")
    summary = (
        f"Qualia's decoded brain response is led by {top[0]['plain'].lower()}, "
        f"{top[1]['plain'].lower()}. A directional read of how the brain responds, "
        "not a clinical scan."
        if len(top) >= 2 else
        "A directional read of how the brain responds, not a clinical scan."
    )

    network_ts = {name: [r6(net_per_sec[s, net_idx[name]]) for s in range(n_sec)] for name in YEO7_ORDER}

    return {
        "capability": "brain_map",
        "modality": "video",
        "model": "qualia",
        "headline": headline,
        "summary": summary,
        "score": overall,
        "networks": networks,
        "topRegions": top_regions,
        "durationSec": n_sec,
        # mediaUrl is the PLAYABLE clip (stored kairo-uploads signed URL, or a
        # direct media file); sourceUrl keeps the original page for attribution.
        "mediaUrl": media_url or art.get("source_url"),
        "sourceUrl": art.get("source_url"),
        "kpiTimeSeries": {k: [r6(v) for v in kpi_ts[k].tolist()] for k in KPI_NAMES},
        "networkTimeSeries": network_ts,
        "kpiSummary": kpi_summary,
        "kpiPeaks": None,
        "composites": composites,
        "verdictSummary": {"score": overall, "grade": grade, "peak_multiplier": peak_mult},
        # Audit: which kairo-serve streams were zero-filled at score time.
        "qualia_zero_filled": (art.get("extract_manifest") or {}).get("zero_filled"),
        "qualia_real_streams": (art.get("extract_manifest") or {}).get("real"),
        "qualia_meta": art.get("meta"),
        # Spoken transcript (whisper, produced for the Llama stream) — surfaced so the
        # web grounded "why" + chat can cite real content. None when silent/no speech.
        "transcript": (art.get("transcript") or {}).get("text") or None,
        "transcriptSegments": (art.get("transcript") or {}).get("segments") or None,
        "transcriptLanguage": (art.get("transcript") or {}).get("language") or None,
    }


# ── Persist the downloaded clip so the verdict can PLAY + scrub it ────────────
# Mirrors mary/modal/serve.py::_store_playable: upload the bytes to the shared
# kairo-uploads bucket and return a long-lived signed URL. The clip was staged
# on the /extracted volume by kairo-extract.extract_all (clip_path); we read it
# here because THIS app is the one holding the Supabase service-role creds.
UPLOAD_BUCKET = "kairo-uploads"


def _store_playable(sb, clip_path: str, run_id: str) -> str | None:
    """Read the staged clip off the /extracted volume, upload it to
    kairo-uploads/qualia/<run_id><ext>, and return a 1-year signed URL the
    verdict media frame can play + scrub. Best-effort: any failure → None and
    the UI falls back to its cover/poster state."""
    import os as __os
    try:
        if not clip_path or not __os.path.exists(clip_path):
            print(f"[qualia-serve] _store_playable: clip not found at {clip_path!r}")
            return None
        ext = __os.path.splitext(clip_path)[1] or ".mp4"
        key = f"qualia/{run_id}{ext}"
        with open(clip_path, "rb") as f:
            data = f.read()
        ctype = ("video/mp4" if ext in (".mp4", ".m4v", ".mov") else
                 "video/webm" if ext == ".webm" else
                 "application/octet-stream")
        try:
            sb.storage.from_(UPLOAD_BUCKET).upload(
                key, data, {"content-type": ctype, "upsert": "true"})
        except Exception:
            # already exists / older client signature — try update, then re-sign
            sb.storage.from_(UPLOAD_BUCKET).update(
                key, data, {"content-type": ctype})
        res = sb.storage.from_(UPLOAD_BUCKET).create_signed_url(key, 31536000)
        if isinstance(res, dict):
            return res.get("signedURL") or res.get("signedUrl") or res.get("signed_url")
        return None
    except Exception as e:
        print(f"[qualia-serve] _store_playable failed: {e!r}")
        return None


@app.function(image=image, secrets=SECRETS, volumes=VOLUMES, timeout=7200)
def run_job(run_id: str, input_url: str, subject: int = 0, with_internvl: bool = True) -> dict:
    """Async Qualia job — the analog of mary-serve's run_job. Writes status +
    the finished Mary-shaped artifact back into mary_runs, EXACTLY like
    mary-serve, so the frontend's submit→poll flow + the API completion path work
    unchanged. Spawned by `submit`; never blocks the Express handler."""
    import datetime as _dt
    from supabase import create_client

    sb = create_client(_os.environ["SUPABASE_URL"], _os.environ["SUPABASE_SERVICE_ROLE_KEY"])

    def now():
        return _dt.datetime.now(_dt.timezone.utc).isoformat()

    def update(**fields):
        try:
            sb.table("mary_runs").update(fields).eq("id", run_id).execute()
        except Exception as e:  # never let a status write kill the job
            print(f"[qualia-serve] mary_runs update failed for {run_id}: {e!r}")

    try:
        update(status="extracting", progress=20, started_at=now())
        # Full extract (download → 18 streams) + score → Mary-shaped artifact.
        extract_all = modal.Function.from_name("kairo-extract", "extract_all")
        manifest = extract_all.remote(input_url, with_internvl)
        update(status="predicting", progress=60)
        art = Kairo().score_npz.remote(manifest["npz"], subject, manifest.get("npz_bytes"))
        art["source_url"] = input_url
        art["transcript"] = manifest.get("transcript")  # spoken transcript (or None)
        art["extract_manifest"] = {
            "real": manifest.get("real"), "zero_filled": manifest.get("zero_filled"),
            "T": manifest.get("T"), "npz": manifest.get("npz"),
        }

        # ── Playback persistence ──────────────────────────────────────────────
        # Mirror mary-serve: hand the verdict a STORED, playable clip URL rather
        # than the raw page link (instagram/tiktok/youtube pages can't load in a
        # <video>). If the input was ALREADY a direct, playable media file we
        # don't re-store it — just pass it straight through. Otherwise we mirror
        # the clip kairo-extract staged on /extracted into kairo-uploads and sign
        # it. `source_url` (set above) preserves the original page for attribution.
        media_url = input_url
        path_lower = input_url.split("?", 1)[0].lower()
        is_direct_media = path_lower.endswith(
            (".mp4", ".mov", ".m4v", ".webm")) or "/storage/v1/object/" in input_url
        if not is_direct_media:
            try:
                extracted_vol.reload()  # see the clip extract_all just committed
            except Exception:
                pass
            stored = _store_playable(sb, manifest.get("clip_path"), run_id)
            media_url = stored or input_url  # fall back to page url if storage failed
        artifact = build_qualia_artifact(art, media_url)
        n_sec = artifact.get("durationSec") or 0
        update(status="complete", progress=100, result=artifact,
               duration_sec=n_sec, completed_at=now())
        print(f"[qualia-serve] run {run_id} complete (n_sec={n_sec}, "
              f"zero_filled={len((art['extract_manifest'] or {}).get('zero_filled') or [])})")
        return {"ok": True, "run_id": run_id, "score": artifact.get("score")}
    except Exception as e:
        update(status="error", error=(str(e) or repr(e) or "qualia analysis failed"),
               error_step="run_job", completed_at=now())
        print(f"[qualia-serve] run_job error {run_id}: {e!r}")
        return {"ok": False, "run_id": run_id, "error": str(e)}


@app.function(image=image, secrets=SECRETS, timeout=60)
@modal.fastapi_endpoint(method="POST")
def submit(payload: dict) -> dict:
    """Express → Modal trigger (mirrors mary-serve's submit). Verifies the shared
    pipeline secret, spawns the async Qualia job, and returns the modal call id
    immediately. The job writes back to mary_runs — so this returns in <1s and the
    serverless handler never times out (the 502 root cause)."""
    from fastapi import HTTPException
    expected = _os.environ.get("MARY_PIPELINE_SECRET")
    if not expected or payload.get("auth_token") != expected:
        raise HTTPException(status_code=401, detail="bad auth_token")
    run_id = payload.get("run_id")
    input_url = payload.get("input_url")
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id required")
    if not input_url:
        raise HTTPException(status_code=400, detail="input_url required (Qualia scores a video URL)")
    call = run_job.spawn(
        run_id=run_id, input_url=input_url,
        subject=int(payload.get("subject", 0) or 0),
        # Visual ON by default: run the full InternVL3 + V-JEPA(L) + VideoMAEv2 +
        # V-JEPA2 visual front-end. Caller can pass with_internvl=false to skip the
        # one expensive backbone.
        with_internvl=bool(payload.get("with_internvl", True)),
    )
    return {"modal_call_id": call.object_id}


@app.local_entrypoint()
def main(mode: str = "gate", clip_a: str = "s01e01a", clip_b: str = "s01e15a", subject: int = 0,
         url_a: str = "", url_b: str = "", with_internvl: bool = False, skip_visual: bool = False):
    import json
    if mode == "gate_url":
        print(json.dumps(gate_url.remote(url_a, url_b, subject, with_internvl, skip_visual), indent=2, default=str))
        return
    if mode == "score_url":
        print(json.dumps(score_url.remote(url_a or clip_a, subject, with_internvl), indent=2, default=str)[:3000])
        return
    if mode == "gate":
        print(json.dumps(gate.remote(clip_a, clip_b, subject), indent=2, default=str))
    elif mode == "score":
        print(json.dumps(score_clip.remote(clip_a, subject), indent=2, default=str)[:2000])
    else:
        raise SystemExit("mode must be 'gate' or 'score'")

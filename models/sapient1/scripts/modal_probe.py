"""Stage-2 Modal probe — cheapest possible validation of the Modal stack.

Spins up a tiny A100-40GB function that:
  1. Loads the Modal Docker image (verifies image build succeeds).
  2. Mounts `/data` from the shared Modal Volume `sapient-data` (verifies
     create_if_missing=True works and volume.commit() round-trips).
  3. Imports the `hf-token` Modal secret as HUGGINGFACE_HUB_TOKEN env.
  4. Confirms CUDA + GPU + VRAM are visible inside the container.
  5. Attempts to download `meta-llama/Llama-3.2-3B/config.json` from HF
     using the injected token — this proves the gated-model access
     pipeline works in production identical to local.

What this does NOT do:
  • No model weight downloads (only the config.json ~1 KB).
  • No encoder forward pass.
  • No long-running compute.

Expected cost: A100-40GB at ~$1.50/hr × ~3 min total (mostly Docker cold-start
on the first run) ≈ $0.05–0.20. Subsequent runs reuse the cached image and
should be under $0.02.

Run:
  modal run scripts/modal_probe.py

Cleanup (after the probe succeeds):
  modal volume rm sapient-data --confirm    # if you want to start fresh
"""

from __future__ import annotations

import modal

APP_NAME = "sapient-1-modal-probe"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "huggingface_hub>=0.25,<2",
    )
)

volume = modal.Volume.from_name("sapient-data", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=300,
    volumes={"/data": volume},
    secrets=[hf_secret],
)
def probe() -> dict:
    import os
    import platform
    from pathlib import Path

    import torch

    report: dict = {"app": APP_NAME, "ok": True, "checks": {}}

    # IMPORTANT: every value below is cast to a stdlib type (str/int/bool/
    # float/list) so the pickled result deserializes locally even when the
    # caller has no torch installed (e.g. the pipx-installed modal CLI runs
    # in a Python venv with only modal + stdlib). Subtle types like
    # torch.version.TorchVersion carry their module ref into pickle and
    # break deserialization on import-less clients.

    # 1. Container basics
    report["checks"]["python"] = str(platform.python_version())
    report["checks"]["torch"]  = str(torch.__version__)

    # 2. GPU
    report["checks"]["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        report["checks"]["gpu_name"] = str(torch.cuda.get_device_name(0))
        report["checks"]["gpu_vram_gb"] = float(round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1
        ))
    else:
        report["ok"] = False
        report["checks"]["gpu_error"] = "torch.cuda.is_available() returned False"

    # 3. /data mount
    data_root = Path("/data")
    report["checks"]["data_mounted"] = bool(data_root.exists())
    if data_root.exists():
        marker = data_root / ".modal-probe-ok"
        marker.write_text("ok\n")
        volume.commit()
        report["checks"]["data_write_ok"] = bool(marker.exists())

    # 4. HF token + gated-model access
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    report["checks"]["hf_token_present"] = bool(token)
    if token:
        from huggingface_hub import hf_hub_download
        try:
            cfg_path = hf_hub_download(
                "meta-llama/Llama-3.2-3B",
                filename="config.json",
                token=token,
            )
            report["checks"]["llama_gated_ok"] = True
            report["checks"]["llama_config_size_bytes"] = int(Path(cfg_path).stat().st_size)
        except Exception as e:
            report["ok"] = False
            report["checks"]["llama_gated_error"] = str(repr(e))

    # 5. HF org membership (does the token see The-Sapient-Company?)
    if token:
        import urllib.request
        try:
            req = urllib.request.Request(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {token}"},
            )
            import json as _json
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = _json.loads(resp.read())
            orgs = [str(o["name"]) for o in data.get("orgs", [])]
            report["checks"]["hf_user"] = str(data.get("name", ""))
            report["checks"]["hf_orgs"] = orgs
            report["checks"]["hf_in_target_org"] = bool("The-Sapient-Company" in orgs)
            if "The-Sapient-Company" not in orgs:
                report["ok"] = False
        except Exception as e:
            report["ok"] = False
            report["checks"]["hf_whoami_error"] = str(repr(e))

    print("=== Modal probe report ===")
    for k, v in report["checks"].items():
        print(f"  {k}: {v}")
    print(f"  OVERALL: {'✅ PASS' if report['ok'] else '⚠️  FAIL'}")
    return report


@app.local_entrypoint()
def main() -> None:
    """`modal run scripts/modal_probe.py`"""
    result = probe.remote()
    if result["ok"]:
        print("\n✅ Modal stack is healthy. Stage 2 complete.")
    else:
        print("\n⚠️  At least one check failed — see report above.")
        raise SystemExit(1)

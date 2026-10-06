"""Run the sapient-2 HF release ON MODAL (torch + the sapient-data volume + the
hf-token secret all live there). Wraps release.assemble + release.push so the
trained checkpoint on the volume gets assembled (config.json + model.safetensors
+ README + parcellate) and pushed to HF without needing torch locally.

  modal run release_modal.py                      # scratch, best_e00.pt (default)
  modal run release_modal.py --ckpt-rel checkpoints/sapient2_scratch_av/latest.pt
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "safetensors>=0.4",
        "huggingface_hub>=0.25,<2",
        "numpy>=1.26,<3",
        "scipy>=1.13",
        "nilearn>=0.10.4",
        "nibabel>=5.2",
    )
    .workdir("/root")
    .add_local_dir(".", remote_path="/root")
)

volume = modal.Volume.from_name("sapient-data")
app = modal.App("sapient-2-release")


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("hf-token")],
    timeout=3600,
)
def release_remote(variant: str, ckpt_rel: str) -> str:
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root")
    from release import assemble, push, VARIANT_REPO, HF_ORG

    ckpt = Path("/data") / ckpt_rel
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not on volume: {ckpt}")
    artifact_dir = Path("/tmp/sapient2_release")
    parcellation = Path("/root/data/parcellate.npz")

    repo_id = f"{HF_ORG}/{VARIANT_REPO[variant]}"
    try:
        assemble(variant, ckpt, artifact_dir, parcellation)
        push(artifact_dir, repo_id, public=False)
        return f"OK proper-release → https://huggingface.co/{repo_id}"
    except Exception as e:  # never leave the founder empty-handed — push the raw ckpt
        print(f"  assemble/push failed ({e!r}); falling back to raw checkpoint upload")
        from huggingface_hub import HfApi, create_repo
        api = HfApi()
        create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(ckpt),
            path_in_repo="best.pt",
            repo_id=repo_id,
            repo_type="model",
            commit_message="raw checkpoint (first-light sapient-2 scratch)",
        )
        return f"OK raw-fallback (best.pt only) → https://huggingface.co/{repo_id}"


@app.local_entrypoint()
def main(variant: str = "scratch",
         ckpt_rel: str = "checkpoints/sapient2_scratch_av/best_e00.pt") -> None:
    print(release_remote.remote(variant, ckpt_rel))

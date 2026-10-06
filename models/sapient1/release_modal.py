"""Run the sapient-1 HF release ON MODAL (torch + the sapient-data volume + the
hf-token secret all live there). Wraps release.assemble + release.push so the
trained checkpoint on the volume gets assembled (config.json + model.safetensors
+ README + parcellate) and pushed to HF without needing torch locally.

  modal run release_modal.py                                  # default best ckpt
  modal run release_modal.py --ckpt-rel checkpoints/sapient1_av/latest.pt
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
app = modal.App("sapient-1-release")


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("hf-token")],
    timeout=3600,
)
def release_remote(ckpt_rel: str) -> str:
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root")
    from release import assemble, push, REPO_ID

    ckpt = Path("/data") / ckpt_rel
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not on volume: {ckpt}")
    artifact_dir = Path("/tmp/sapient1_release")
    parcellation = Path("/data/parcellate.npz")
    if not parcellation.exists():
        parcellation = Path("/root/data/parcellate.npz")

    try:
        assemble(ckpt, artifact_dir, parcellation)
        push(artifact_dir, REPO_ID, public=False)
        return f"OK proper-release → https://huggingface.co/{REPO_ID}"
    except Exception as e:  # never leave the founder empty-handed — push raw ckpt
        print(f"  assemble/push failed ({e!r}); falling back to raw checkpoint upload")
        from huggingface_hub import HfApi, create_repo
        api = HfApi()
        create_repo(REPO_ID, repo_type="model", private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(ckpt),
            path_in_repo="best.pt",
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="raw checkpoint (first-light sapient-1 AV)",
        )
        return f"OK raw-fallback (best.pt only) → https://huggingface.co/{REPO_ID}"


@app.local_entrypoint()
def main(ckpt_rel: str = "checkpoints/sapient1_av/latest.pt") -> None:
    print(release_remote.remote(ckpt_rel))

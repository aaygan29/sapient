"""Push the trained Mary checkpoint(s) from the Modal volume to HuggingFace.
Branding: **Mary** only — no ORCLE / no reference-paper names.

By default this points at the SERVED single-seed weights
`/data/checkpoints/mary_multi_stable_s13/best.pt` (what MaryServer actually loads).
Both the checkpoint source variant and the seed list are overridable so we can
re-point at the new responsive ensemble `mary_multi_stable_resp_s{13,17,23}` once
trained, without editing this file:

  # served single-seed model (default)
  modal run scripts/push_to_hf.py --repo The-Sapient-Company/mary

  # the new responsive 3-seed ensemble (after training)
  modal run scripts/push_to_hf.py --repo The-Sapient-Company/mary \\
      --variant mary_multi_stable_resp --seeds 13,17,23
"""
import modal

app = modal.App("mary-hf-push")
vol = modal.Volume.from_name("sapient-data")
image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub>=0.25,<2")
hf_secret = modal.Secret.from_name("hf-token")

MODEL_CARD = """---
license: cc-by-nc-4.0
library_name: pytorch
tags: [brain-encoding, fmri, multimodal, neuroscience]
---

# Mary — Multimodal Brain Encoder

Mary predicts fMRI BOLD responses across 20,484 fsaverage5 cortical vertices from naturalistic
audio / video / text stimuli. It pairs six frozen multimodal foundation-model backbones (motion, scene,
environmental audio, speech, narrative text, on-screen text) with a lightweight trainable adapter
(~112M params: per-stream HRF convolution, cross-attention fusion, attentive temporal pooling, RoPE
prediction transformer) over the FROZEN cached backbone features, with **group + per-subject
prediction heads**.

**Architecture:** d_model 1024, 2 fusion + 4 prediction layers, 8 heads. The adapter is trained over
cached frozen features; the backbones are not fine-tuned.

**Released weights:** the single-seed served model `mary_multi_stable_s13`, trained on open
CC0/CC-BY naturalistic-fMRI datasets (Huth Narratives, Lebel2023, HAD, CNeuroMod, Wen2017) with the
proven stable recipe (LR 1e-4, correlation-led loss). When the responsive ensemble is published this
card is re-pointed at `mary_multi_stable_resp_s{13,17,23}` (per-vertex softmax-combinable).

**Held-out performance (proof-of-concept scale):** the loss and checkpoint selection are
correlation-led. Note that ~90% of the 20,484 vertices are noise (ISC≈0), so a whole-brain mean
Pearson r understates the real signal ~10:1; we report both:
- Whole-brain mean vertex r ≈ **0.028**
- Responsive vertices (ISC > 0.05) mean vertex r ≈ **0.034**

These are early, small-open-data magnitudes — they are not state-of-the-art claims. See the Mary
technical reports for full methodology, noise ceilings, Yeo-7 decompositions, and limitations.

- Output space: 20,484 fsaverage5 vertices
- Inputs: audio (speech/events), text (narrative), video (where available)

© The Sapient Company.
"""


@app.function(image=image, timeout=3600, volumes={"/data": vol}, secrets=[hf_secret])
def push(repo: str, variant: str, seeds: list[int]):
    import os
    from pathlib import Path
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    # model card
    Path("/tmp/README.md").write_text(MODEL_CARD)
    api.upload_file(path_or_fileobj="/tmp/README.md", path_in_repo="README.md", repo_id=repo)
    # checkpoints — source the SERVED weights from the Modal volume (renamed Mary-only).
    pushed = []
    for s in seeds:
        src = Path(f"/data/checkpoints/{variant}_s{s}/best.pt")
        if src.exists():
            api.upload_file(path_or_fileobj=str(src),
                            path_in_repo=f"mary_s{s}/best.pt", repo_id=repo)
            pushed.append(f"mary_s{s}")
        else:
            print(f"  skip: {src} not found")
    print(f"Pushed to https://huggingface.co/{repo} : {pushed}")
    return {"repo": repo, "variant": variant, "pushed": pushed}


@app.local_entrypoint()
def main(repo: str = "The-Sapient-Company/mary",
         variant: str = "mary_multi_stable",
         seeds: str = "13"):
    """Default: served single-seed `mary_multi_stable_s13/best.pt`.

    Re-point at the responsive ensemble with:
      --variant mary_multi_stable_resp --seeds 13,17,23
    """
    seed_list = [int(x) for x in seeds.split(",") if x.strip()]
    print(push.remote(repo, variant, seed_list))

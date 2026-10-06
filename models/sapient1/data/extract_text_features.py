"""Extract Llama-3.2-3B word-level features, frozen.

Per spec §2.3 and §13. Llama-3.2-3B is the SOLE text encoder for v1 (Qwen
removed 2026-05-26). hidden_dim=3072. The codebase is dimension-agnostic;
swapping in a different text encoder later is a config change.

For each word in the transcript:
  1. Look at preceding 1024-word context (window ending at the current word).
  2. Tokenize the context and run a single forward pass.
  3. Average token embeddings for the tokens belonging to the current word.
  4. Time-align to the word's onset (from forced alignment).
After all words: resample to a 2 Hz evenly-spaced grid → (N_STEPS, 3072).

We expect each stimulus to ship with a sibling JSON of the form:
    <stem>.words.json = [{"word": "the", "onset": 0.10, "offset": 0.28}, ...]
If forced alignment is missing, generate it offline with whisper-timestamped
or aeneas and re-run. We DO NOT cache audio inside this script.

Modal app: sapient-1-features-llama (A100-40GB per spec §12).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import modal

APP_NAME = "sapient-1-features-llama"
MODEL_ID = "meta-llama/Llama-3.2-3B"
HIDDEN_DIM = 3072
CONTEXT_WORDS = 1024
TARGET_RATE_HZ = 2.0


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.46.0",
        "sentencepiece==0.2.0",
        "tokenizers==0.20.0",
        "numpy>=1.26,<3",
        "huggingface_hub>=0.25,<2",
        "safetensors>=0.4",
    )
)

volume = modal.Volume.from_name("sapient-data", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

app = modal.App(APP_NAME)


def _load_word_timing(words_json: Path) -> list[dict]:
    with words_json.open() as f:
        return json.load(f)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=24 * 60 * 60,
    volumes={"/data": volume},
    secrets=[hf_secret],
)
def extract(stim_root: str, out_root: str) -> None:
    import os
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    stim_dir = Path(stim_root)
    out_dir = Path(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = (
        AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        .eval()
        .cuda()
    )

    word_jsons = sorted(stim_dir.rglob("*.words.json"))
    print(f"Found {len(word_jsons)} transcripts under {stim_dir}.")

    for words_path in word_jsons:
        rel = words_path.relative_to(stim_dir).with_suffix("")
        # Drop the trailing '.words' from the stem
        rel = rel.with_name(rel.name.removesuffix(".words")).with_suffix(".npy")
        out = out_dir / rel
        if out.exists():
            print(f"  skip {out}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)

        print(f"  extract {words_path} → {out}")
        words = _load_word_timing(words_path)
        if not words:
            print("    empty transcript; skipping")
            continue

        duration_s = float(words[-1]["offset"])
        per_word_emb = np.empty((len(words), HIDDEN_DIM), dtype=np.float32)

        for i, w in enumerate(words):
            ctx_start = max(0, i - CONTEXT_WORDS + 1)
            ctx_words = words[ctx_start : i + 1]
            text = " ".join(x["word"] for x in ctx_words)
            enc = tok(text, return_tensors="pt", add_special_tokens=True)
            input_ids = enc.input_ids.cuda()
            with torch.no_grad():
                hidden = model(input_ids).last_hidden_state.squeeze(0)  # (T, 3072)

            # Find the token range belonging to the FINAL word (the "current" word).
            # Strategy: tokenize the context without the final word, then anything
            # added afterwards is the final word's tokens.
            prefix = " ".join(x["word"] for x in ctx_words[:-1])
            prefix_ids = tok(prefix, return_tensors="pt", add_special_tokens=True).input_ids[0]
            n_prefix = prefix_ids.shape[0]
            # Trailing tokens (excluding any final EOS) are the current word.
            cur_token_emb = hidden[n_prefix:].float()
            if cur_token_emb.shape[0] == 0:
                # Degenerate (e.g. tokenizer produced no extra tokens) — fall
                # back to the last hidden state.
                cur_token_emb = hidden[-1:].float()
            per_word_emb[i] = cur_token_emb.mean(dim=0).to("cpu").numpy()

        # Resample word-level (irregular) → 2 Hz evenly-spaced grid.
        # Per-step value = mean of all words whose midpoint falls in [t, t+0.5s).
        n_steps = int(duration_s * TARGET_RATE_HZ)
        feats = np.zeros((n_steps, HIDDEN_DIM), dtype=np.float16)
        if n_steps > 0:
            midpoints = np.array(
                [0.5 * (w["onset"] + w["offset"]) for w in words]
            )
            bin_idx = np.minimum(
                (midpoints * TARGET_RATE_HZ).astype(np.int64),
                n_steps - 1,
            )
            counts = np.zeros(n_steps, dtype=np.int64)
            for emb, b in zip(per_word_emb, bin_idx):
                feats[b] += emb.astype(np.float16)
                counts[b] += 1
            # Avoid div-by-zero: bins with no word retain zeros and will be
            # forward-filled at training time by the dataset's HRF window.
            nonempty = counts > 0
            feats[nonempty] = (
                feats[nonempty].astype(np.float32) / counts[nonempty, None]
            ).astype(np.float16)

        np.save(out, feats)
        volume.commit()
    print("Done.")


@app.local_entrypoint()
def main(
    stim_root: str = "/data/raw/cneuromod/fmriprep/friends/sourcedata/friends/stimuli",
    out_root: str = "/data/features/llama/cneuromod",
) -> None:
    extract.remote(stim_root, out_root)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stim-root", required=True)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()
    main(args.stim_root, args.out_root)

# sapient-1 — Architecture

10-minute read for a new engineer. Pulls the relevant slices out of [`../engineering-outline/08_sapient1_sapient2_spec.md`](../engineering-outline/08_sapient1_sapient2_spec.md).

## 1. Goal

Predict whole-cortex fMRI activity at 20,484 fsaverage5 vertices in response to multimodal stimulus (video, audio, text) sampled at 1 Hz, given a 100-second window of stimulus features.

## 2. Data flow (single training step)

```
Video clip → V-JEPA 2 Gigantic (frozen) → (200, 1280) features at 2 Hz
Audio clip → Wav2Vec-BERT 2.0 (frozen)  → (200, 1024) features at 2 Hz
Transcript → Llama-3.2-3B (frozen)      → (200, 3072) features at 2 Hz
                                                  │
                                                  ▼
                            ┌─── proj_video : Linear(1280 → 1152)
                            ├─── proj_audio : Linear(1024 → 1152)
                            └─── proj_text  : Linear(3072 → 1152)
                                                  │
                                  modality dropout (p=0.3) on each
                                                  │
                                      SUM the 3 projections
                                                  │
                                  Mean-pool pairs: 2 Hz → 1 Hz
                                                  │
                                       + pos_embed (learnable)
                                       + subject_embed (dropout p=0.1)
                                                  │
                                 TransformerEncoder × 8 layers,
                                 hidden=1152, heads=8, ff=4608,
                                 pre-norm, GELU
                                                  │
                                low_rank : Linear(1152 → 2048, no bias)
                                                  │
                                brain_head : Linear(2048 → 20484)
                                                  │
                                                  ▼
                       fMRI prediction (B, 100, 20484)
```

## 3. The 15 architecture decisions (locked)

Verbatim from [`../engineering-outline/00_HANDOFF_README.md`](../engineering-outline/00_HANDOFF_README.md) "Architecture lock". If you ever feel the urge to deviate, cite the decision number and ask first.

1. **Modality dropout** `p=0.3` applied **after projections, before sum**.
2. **Subject dropout** `p=0.1` at training time only.
3. **Pos embed:** `nn.Parameter(1, 100, 1152) × 0.02`, **learnable**, `max_len=100`.
4. **Pos embed + subject embed** added **after** the 2 Hz → 1 Hz mean-pool.
5. **Order:** project → modality_dropout → SUM → pool → +pos → +subj → transformer.
6. **Subject embed** broadcast: `subject_embed[idx].unsqueeze(1)` over 100 timesteps.
7. **Two-stage head:** `low_rank: Linear(1152 → 2048, bias=False)` then `brain_head: Linear(2048 → 20484)`.
8. **HRF offset:** fMRI window = `[start_tr+5, start_tr+105)`, stim window = `[start_tr*2, (start_tr+100)*2)`.
9. **Whisper-large-v3** is an ablation only. **W2V-BERT 2.0** is the primary audio encoder.
10. **Schaefer-1000 atlas** is built once offline, used only in post-processing.
11. **sapient-2-ft** uses `lr=5e-5`, not `1e-5`.
12. **sapient-2-ft** drops `subject_embed` when loading sapient-1 weights.
13. **n_subjects:** 4 for sapient-1; ~75 for sapient-2.
14. **BOLD Moments + Narratives = augmentation only** (week 2). **CNeuroMod = primary** (week 1).
15. **Loss:** per-vertex squared error → mean over 20,484 vertices → mask-weighted mean over time.

## 4. Training loop summary

- AdamW, lr=1e-4, weight_decay=0.0, OneCycleLR with pct_start=0.1
- Batch size 8/GPU, grad_accum=2 → effective batch 16
- bfloat16 mixed precision (A100/H100 native)
- Gradient clip at 1.0
- 15 epochs target; 24h on H100-80GB realistically completes 3–5 epochs on the CNeuroMod subset

## 5. Eval

- **Vertex-level Pearson r:** for each of 20,484 vertices, correlate prediction vs ground truth across all held-out timepoints. Report mean + top-10% mean.
- **Schaefer-1000 parcel Pearson:** project predictions through the offline-built `parcellate.npz` (sparse 1000×20484 averaging matrix), then correlate per parcel.
- **Brain surface plot:** nilearn `plot_surf_stat_map` of vertex Pearson on fsaverage5. Visual cortex should be hottest.

Acceptance threshold: vertex Pearson mean > 0.10 on held-out (sanity); > 0.20 is genuinely strong for single-GPU.

## 6. What we did NOT build

- No fine-tune of TRIBE v2 weights anywhere in the lineage.
- No source code copied from `facebookresearch/algonauts-2025`.
- Llama-3.2-3B is the only text encoder — Qwen / Mistral / Gemma deliberately out of scope for v1.

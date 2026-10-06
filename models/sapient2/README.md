# sapient-2

A conversation-specialized brain encoder. Same architecture as `sapient-1`, retrained on human-human and human-robot interaction fMRI corpora. Two training lineages ship as separate HF releases:

- **`sapient-2-scratch-llama`** — random init, trained from scratch on ds004996 + ds001740.
- **`sapient-2-ft-llama`** — initialized from `sapient-1-llama`, fine-tuned at `lr=5e-5`. The `subject_embedding` table is dropped on load (different subject set).

We benchmark both and ship the higher-correlation variant; both are released for reproducibility.

**Status:** scaffold only. Architecture, training loop, and eval to be implemented per [`../engineering-outline/08_sapient1_sapient2_spec.md`](../engineering-outline/08_sapient1_sapient2_spec.md). HF releases `The-Sapient-Company/sapient-2-{scratch,ft}-llama` are **not yet uploaded**.

## At a glance

- Same 8-layer, hidden=1152, ~180M-parameter transformer as sapient-1
- Frozen feature encoders: V-JEPA 2 Gigantic (video), Wav2Vec-BERT 2.0 (audio), Llama-3.2-3B (text)
- Output: 20,484 fsaverage5 cortical-surface vertices @ 1 Hz
- Training corpus: ds004996 NeuroEngage primary (~16.5 h, ~50 subjects), ds001740 Rauchbauer HRI augmentation (~10 h, 25 subjects). Pooled ~26.5 h, ~75 subjects.
- License: Apache 2.0; upstream encoder licenses preserved (see [`NOTICE`](./NOTICE))
- Built with Llama.

## Quickstart

```bash
make setup            # install deps, datalad install ds004996 + ds001740
make features         # extract V-JEPA + W2V-BERT + Llama features (Modal)
make train-scratch    # train sapient-2-scratch-llama from random init
make train-ft         # train sapient-2-ft-llama from sapient-1-llama init
make eval             # vertex + parcel Pearson eval, brain-surface plot
make release          # build release_artifacts/, upload to HF
make test             # pytest tests/
```

See [`Makefile`](./Makefile) for what each target actually runs.

## Read first

1. [`ARCHITECTURE.md`](./ARCHITECTURE.md) — how the model works (same as sapient-1).
2. [`CODEBASE_MAP.md`](./CODEBASE_MAP.md) — file-by-file walkthrough.
3. [`../engineering-outline/08_sapient1_sapient2_spec.md`](../engineering-outline/08_sapient1_sapient2_spec.md) — full build spec.
4. [`../engineering-outline/05_neuroengage_finetune_walkthrough.md`](../engineering-outline/05_neuroengage_finetune_walkthrough.md) — fine-tune runbook (if present).

## License

This model: Apache 2.0 (see [`LICENSE`](./LICENSE)).

The frozen Llama-3.2-3B text encoder is subject to the [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/). V-JEPA 2 and Wav2Vec-BERT 2.0 are MIT. The `sapient-2-ft` variant is fine-tuned from `sapient-1-llama` (also Apache 2.0). See [`NOTICE`](./NOTICE) for the full third-party attribution.

## Citation

See [`CITATION.cff`](./CITATION.cff).

# sapient-1

A trimodal brain encoder predicting whole-cortex fMRI response to video, audio, and text stimuli.

**Status:** scaffold only. Architecture, training loop, and eval to be implemented per [`../engineering-outline/08_sapient1_sapient2_spec.md`](../engineering-outline/08_sapient1_sapient2_spec.md). HF release `The-Sapient-Company/sapient-1-llama` is **not yet uploaded**.

## At a glance

- 8-layer transformer, hidden=1152, 8 attention heads, ~180M trainable parameters
- Frozen feature encoders: V-JEPA 2 Gigantic (video), Wav2Vec-BERT 2.0 (audio), Llama-3.2-3B (text)
- Output: 20,484 fsaverage5 cortical-surface vertices @ 1 Hz
- Training corpus: CNeuroMod 4 CC0 subjects (sub-01, 02, 03, 05) primary; BOLD Moments + Narratives augmentation
- License: Apache 2.0; upstream encoder licenses preserved (see [`NOTICE`](./NOTICE))
- Built with Llama.

## Quickstart

```bash
make setup     # install deps, datalad install CNeuroMod CC0 subset
make features  # extract V-JEPA + W2V-BERT + Llama features (Modal jobs)
make train     # train sapient-1-llama on H100-80GB via Modal
make eval      # vertex + Schaefer-1000 parcel Pearson, brain-surface plot
make release   # build release_artifacts/, upload to HF
make test      # pytest tests/
```

See [`Makefile`](./Makefile) for what each target actually runs.

## Read first

1. [`ARCHITECTURE.md`](./ARCHITECTURE.md) — how the model works.
2. [`CODEBASE_MAP.md`](./CODEBASE_MAP.md) — file-by-file walkthrough.
3. [`../engineering-outline/08_sapient1_sapient2_spec.md`](../engineering-outline/08_sapient1_sapient2_spec.md) — full build spec.

## License

This model: Apache 2.0 (see [`LICENSE`](./LICENSE)).

The frozen Llama-3.2-3B text encoder is subject to the [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/). V-JEPA 2 and Wav2Vec-BERT 2.0 are MIT. See [`NOTICE`](./NOTICE) for the full third-party attribution.

## Citation

See [`CITATION.cff`](./CITATION.cff).

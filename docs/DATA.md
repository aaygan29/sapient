# Datasets and large assets

This repository keeps the code and documentation and deliberately excludes large or license-restricted assets. This file records what they are and how to obtain or regenerate them.

## OpenLAV / LIRIS-ACCEDE videos (excluded)

The `neurobehavioral-mve/openlav/videos/` directory held roughly 1.7 GB of video from the LIRIS-ACCEDE affective-video collection. It is **not** included.

- **Why excluded:** LIRIS-ACCEDE is distributed under an end-user license agreement that restricts redistribution.
- **What is kept:** the ingestion scripts (`experiments/control-ladder/openlav_ingest.py`, `goemotions_ingest.py`) and, where small, the metadata maps and derived feature caches, so the pipeline is reproducible.
- **How to obtain:** request access to LIRIS-ACCEDE from its official source, accept the EULA, then run the ingestion scripts.

## Trained model weights (excluded)

Trained checkpoints such as the TRIBE-v2 weights (`best.ckpt`, ~700 MB) are **not** committed. They are large and not regenerable from this repository alone.

- The encoder code and configs needed to understand and retrain the models are in `models/`.
- If you need a specific checkpoint, keep a copy in separate storage. It is not part of this public repository.

> Note: because the weights live only outside this repository, keep an independent backup before deleting any local copies.

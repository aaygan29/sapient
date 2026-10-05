# Datasets and large assets

This repository deliberately excludes some large or license-restricted assets. This document records what they are and how to obtain or regenerate them.

## OpenLAV / LIRIS-ACCEDE videos (excluded)

The `neurobehavioral-mve/openlav/videos/` directory held roughly 1.7 GB of video from the LIRIS-ACCEDE affective-video collection, obtained for the minimum viable experiment. It is **not** included here.

- **Why excluded:** LIRIS-ACCEDE is distributed under an end-user license agreement that restricts redistribution. The EULA that accompanied the download is kept at `neurobehavioral-mve/liris_access/LIRIS-ACCEDE_EULA.pdf` for reference.
- **What is kept:** the ingestion scripts (`openlav_ingest.py`, `openlav/download_videos.py`), the metadata maps (`bitstream_map.json`, `data_map.json`, `video_data.csv`, `video_database.csv`), and any derived feature caches (`video_feat_cache.npz`). These make the pipeline reproducible.
- **How to obtain:** request access to LIRIS-ACCEDE from its official source, accept the EULA, then run the ingestion scripts to repopulate `openlav/videos/`.

## Model weights (included via Git LFS)

Trained checkpoints such as `neurobehavioral-mve/tribev2_weights/best.ckpt` are tracked with Git LFS because they exceed the normal file-size limit and are not regenerable from this repository alone. Ensure Git LFS is installed before cloning so the real files are fetched rather than pointer stubs:

```bash
git lfs install
git clone <repo-url>
# or, in an existing clone:
git lfs pull
```

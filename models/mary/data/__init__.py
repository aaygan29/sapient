"""Mary data pipeline — fMRI projection, manifest, and torch Dataset.

Mirrors the sapient1 family pattern (vol→fsaverage5 via nilearn vol_to_surf,
1 Hz fMRI grid, z-score per vertex per run, memmap-backed manifest) adapted to
the ds002345 (Huth/Princeton "Narratives") audio-story dataset.
"""

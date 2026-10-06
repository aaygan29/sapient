.PHONY: setup download features forced-align train-scratch train-ft eval release clean test lint typecheck smoke-test status help

PYTHON         := python3.13
VENV           := .venv
SMOKE_VENV     := .venv-smoketest
CONFIG_SCRATCH := configs/sapient2_scratch_llama.yaml
CONFIG_FT      := configs/sapient2_ft_llama.yaml
SMOKE_CONFIG   := configs/sapient2_smoke.yaml

help:
	@echo "sapient-2 Makefile targets:"
	@echo "  make setup         — create local venv, install deps (no data download)"
	@echo "  make download      — pull ds004996 + ds001740 into the sapient-data Modal volume"
	@echo "  make features      — extract V-JEPA + W2V-BERT + Llama features (Modal)"
	@echo "  make forced-align — generate word-level .words.json via whisper-timestamped (Modal)"
	@echo "  make train-scratch — train sapient-2-scratch-llama from random init (H100)"
	@echo "  make train-ft      — train sapient-2-ft-llama from sapient-1-llama init (H100)"
	@echo "  make eval          — vertex + parcel Pearson eval, brain-surface plot"
	@echo "  make release       — assemble release_artifacts/ for both lineages, push to HF"
	@echo "  make test          — pytest tests/"
	@echo "  make lint          — ruff check"
	@echo "  make typecheck     — mypy"
	@echo "  make smoke-test    — end-to-end CPU smoke: synthetic data → train → eval → dry-run release"
	@echo "  make status        — show all sapient-* Modal apps + their state"
	@echo "  make clean         — remove checkpoints/, wandb/, __pycache__/"

setup:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -e ".[dev]"

download:
	# Pulls ds004996 + ds001740 directly into the `sapient-data` Modal volume.
	# Nothing lands on local disk.
	modal run data/download_ds004996.py
	modal run data/download_ds001740.py

features:
	modal run data/extract_video_features.py
	modal run data/extract_audio_features.py
	modal run data/extract_text_features.py

forced-align:
	modal run data/forced_align.py

train-scratch:
	modal run train.py::main_scratch --config $(CONFIG_SCRATCH)

train-ft:
	modal run train.py::main_ft --config $(CONFIG_FT)

eval:
	modal run eval.py --checkpoint $$(ls -t checkpoints/*.pt 2>/dev/null | head -1)

release:
	$(PYTHON) release.py

test:
	. $(VENV)/bin/activate && pytest tests/ -v

lint:
	. $(VENV)/bin/activate && ruff check sapient2/ tests/ data/

typecheck:
	. $(VENV)/bin/activate && mypy sapient2/

status:
	@echo "=== sapient-* Modal apps ==="
	@modal app list 2>&1 | awk -F'│' 'NR<=5 || /sapient-/ {print}' | head -40
	@echo ""
	@echo "=== sapient-data volume contents (top-level) ==="
	@modal volume ls sapient-data / 2>&1 | head -20 || echo "  (volume not yet provisioned)"

smoke-test:
	@if [ ! -d $(SMOKE_VENV) ]; then \
		echo "==> Creating $(SMOKE_VENV) (one-time, ~2 min)"; \
		$(PYTHON) -m venv $(SMOKE_VENV); \
		$(SMOKE_VENV)/bin/pip install -q torch numpy pyyaml scipy pytest safetensors nilearn matplotlib modal huggingface_hub wandb nibabel; \
	fi
	@echo "==> Generate synthetic data"
	$(SMOKE_VENV)/bin/python scripts/make_synthetic_data.py
	@echo "==> Train (1 epoch, smoke config)"
	$(SMOKE_VENV)/bin/python train.py --config $(SMOKE_CONFIG) --local
	@echo "==> Eval on test split"
	$(SMOKE_VENV)/bin/python eval.py --checkpoint $$(ls -t data/checkpoints/sapient2_smoke/best_*.pt 2>/dev/null | head -1) --local --parcellation data/parcellate.npz --out-dir eval
	@echo "==> Release (dry-run, no --confirm)"
	$(SMOKE_VENV)/bin/python release.py --variant scratch --checkpoint $$(ls -t data/checkpoints/sapient2_smoke/best_*.pt | head -1) --parcellation data/parcellate.npz
	@echo ""
	@echo "✅ smoke-test complete."
	@echo "   Inspect: data/checkpoints/sapient2_smoke/ , eval/ , release_artifacts/sapient-2-scratch-llama/"

clean:
	rm -rf checkpoints/ wandb/ runs/ logs/
	rm -rf data/fmri/synthetic data/features data/parcellate.npz data/manifest.json data/checkpoints
	rm -rf eval/ release_artifacts/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +

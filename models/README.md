# sapient-models

The **Sapient brain models** — the encoders that map multimodal video / audio / text stimuli onto human
cortical-surface activation (20,484 fsaverage5 vertices → Yeo-7 networks), plus their **live Modal serving**.
HF releases live under the [`The-Sapient-Company`](https://huggingface.co/The-Sapient-Company) org.

> **Start here:** read [`WORKFLOW.md`](./WORKFLOW.md) — the one source of truth for how research becomes product
> (model → insight → frontend), what to push where, branch/merge rules, and the `handoff/` boundary.

## Folder map (current `main`)
| Folder | What it is | Live? |
|---|---|---|
| [`mary/`](./mary/) | **Mary** — the generalist brain encoder (text/audio/video). Live serving: `mary/modal/serve.py` | ✅ Modal app **`mary-serve`** |
| [`qualia/`](./qualia/) | **Qualia** — the visual model. Live serving: `qualia/modal/serve.py` (deploys `qualia-serve`) + `qualia/serving/extract.py` (deploys `kairo-extract`); plus the engine registry (`core/`), `demo/`, contracts (`QUALIA_CONTRACTS.md`). | ✅ Modal apps **`qualia-serve`** + **`kairo-extract`** |
| [`handoff/`](./handoff/) | The **model ↔ frontend boundary**: `capabilities.json` (the contract) + `samples/` + `CHANGELOG.md` | — |
| [`engineering-outline/`](./engineering-outline/) | Specs, the 15 architecture decisions, licensing, build docs (see read order below) | — |
| `sapient1/`, `sapient2/` | Earlier foundation-encoder generations (base + conversation-specialized). Reference / training lineage. | — |
| `archive/` | Retired material | — |

## `qualia/` holds the live serving (consolidation done 2026-06-10)
- The Qualia serving code now lives under **`qualia/`**: `qualia/modal/serve.py` (was `kairo/kairo_serve.py`, deploys
  `qualia-serve`) and `qualia/serving/{extract,model,rapidapi}.py`. The old top-level `kairo/` folder is **gone**.
- **Internal `kairo` ids are intentionally KEPT as infra:** Modal app `kairo-extract`, the `kairo-*` volumes, and the
  Kairo v4 checkpoint — renaming those risks the live engine, so they stay.
- Old diagnostic probes were archived to `archive/kairo-diagnostics/`.
- Deploy from the new paths: `modal deploy qualia/modal/serve.py` (→ `qualia-serve`) and
  `modal deploy qualia/serving/extract.py` (→ `kairo-extract`).

## Live Modal apps & infra (do not rename)
- Apps: **`mary-serve`** (Mary), **`qualia-serve`** (Qualia, from `qualia/modal/serve.py`), **`kairo-extract`** (feature extractor).
- Volumes keep their `kairo-*` names on purpose: `kairo-weights`, `kairo-features`, `kairo-extracted` (renaming risks the live engine).

## Read order for new engineers
1. [`engineering-outline/00_HANDOFF_README.md`](./engineering-outline/00_HANDOFF_README.md) — orientation + the 15 non-negotiable architecture decisions.
2. [`engineering-outline/PREREQUISITES.md`](./engineering-outline/PREREQUISITES.md) — credentials + tooling readiness checklist.
3. [`engineering-outline/CODEBASE_MAP.md`](./engineering-outline/CODEBASE_MAP.md) — file-by-file walkthrough.
4. [`engineering-outline/09_data_sources_and_licensing.md`](./engineering-outline/09_data_sources_and_licensing.md) — per-dataset license + source URLs.

## Working in this repo
- Work on a `feature/<name>` branch → PR into `main`. **Never push to `main` directly; never deploy on merge.**
- Model/insight changes: add the code **and** declare the capability in `handoff/capabilities.json` with a sample in
  `handoff/samples/` — that's how the frontend picks it up automatically.
- Full rules + the deploy/promote flow: [`WORKFLOW.md`](./WORKFLOW.md).

## Hard rules
- **Llama-3.2-3B is the sole text encoder.** No Qwen, Mistral, Gemma. The `proj_text` Linear is dimension-agnostic; a future swap is one config line.
- **HF org is `The-Sapient-Company`.** Never `simplrads`, never `sapient` (the redirect alias).
- Every model release name ends in `-llama` per the Llama 3.2 Community License §5.
- All training data is **CC0** (CNeuroMod 4 open subjects, ds004996, ds001740) or used features-only (Narratives).

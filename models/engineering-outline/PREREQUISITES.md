# Sapient — Day-0 Prerequisites & Credentials

**Read this BEFORE any code is written.** Every item below must be set up and verified. Claude Code should walk through this list with Robert one item at a time, confirm the credential is in place, and only then proceed to repo scaffolding.

---

## 0. Confirmed by Robert (2026-05-26)

| Item | Status | Notes |
|---|---|---|
| Meta Llama-3.2-3B HF access | ✅ Approved | `meta-llama/Llama-3.2-3B` is available on Robert's account |
| HuggingFace org | ✅ Owned | **`The-Sapient-Company`** (NOT `sapient` — earlier docs say `sapient`; treat `The-Sapient-Company` as canonical and patch on read) |
| Modal H100-80GB quota | ✅ Available | |
| HF write token | ✅ In place | `sapient-claude` (write role, admin on `The-Sapient-Company`); cached at `~/.cache/huggingface/token`; Modal secret `hf-token` refreshed 2026-05-26 |
| datalad + git-annex | ✅ Installed | datalad 1.4.1 (brew), git-annex 10.20260316 (brew) |
| awscli | ✅ Installed | aws-cli/1.45.14 via pipx + python3.13 (brew formula was broken by Python 3.14 pyexpat bug) |
| wandb CLI | ✅ Installed | wandb 0.27.0 via pipx + python3.13. **Local `wandb login` still pending if you plan to run training locally** (Modal jobs already use the existing `wandb` Modal secret) |

---

## 1. HuggingFace setup

**Org name (CANONICAL):** `The-Sapient-Company` → [huggingface.co/The-Sapient-Company](https://huggingface.co/The-Sapient-Company)

**Patch on read:** Anywhere the existing specs say `sapient/<repo>`, it must be read as `The-Sapient-Company/<repo>`. Updated release names:

- `The-Sapient-Company/sapient-1-llama`
- `The-Sapient-Company/sapient-2-scratch-llama`
- `The-Sapient-Company/sapient-2-ft-llama`
- `The-Sapient-Company/{cneuromod,ds004996,ds001740}-features-{vjepa2,w2vbert,llama}` (9 cache repos)
- `The-Sapient-Company/{cneuromod,ds004996,ds001740}-fmri-fsaverage5` (3 fMRI cache repos)
- `The-Sapient-Company/sapient-1-demo`, `The-Sapient-Company/sapient-2-demo` (Spaces)

**Action — update existing code references in CODEBASE_MAP.md §10:**

```python
HF_ORG = "The-Sapient-Company"   # was "sapient"
REPO_ID = f"{HF_ORG}/sapient-1-llama"
```

**Credentials Claude needs from Robert:**

1. **HF token with write access to the org.** Generate at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → "New token" → role: "write" → name it `sapient-build-token`. Scope it to `The-Sapient-Company` org if fine-grained tokens are enabled.
2. Token gets stored as an environment variable: `export HUGGINGFACE_HUB_TOKEN=hf_xxxxx`
3. Also store in Modal secrets (see §2).

**Llama 3.2 gated model:** Already approved — confirm by running:

```bash
hf auth whoami
hf download meta-llama/Llama-3.2-3B --revision main --max-workers 1 --include "config.json"
```

If the `config.json` download succeeds, access is live.

**Documentation links Claude should bookmark:**

- HF Hub Python client: https://huggingface.co/docs/huggingface_hub/index
- `create_repo`, `upload_folder`: https://huggingface.co/docs/huggingface_hub/guides/upload
- Model card YAML schema: https://huggingface.co/docs/hub/model-cards
- Llama 3.2 Community License: https://www.llama.com/llama3_2/license/
- "Built with Llama" attribution guide: https://www.llama.com/docs/help-and-faq/

---

## 2. Modal setup

**Account:** confirm H100-80GB quota is on the Modal workspace tied to Robert's account.

**Credentials Claude needs:**

1. Modal CLI installed: `pip install modal` then `modal setup` (interactive auth)
2. Modal token committed to local `~/.modal.toml`
3. **Modal secrets** to create in the Modal dashboard so apps can pull HF + W&B:
   - `hf-token` → key: `HUGGINGFACE_HUB_TOKEN`, value: the HF write token from §1 (already exists in Robert's Modal workspace as of 2026-05-23; value refreshed 2026-05-26 to the write-scoped `sapient-claude` token)
   - `wandb` → key: `WANDB_API_KEY`, value: from [wandb.ai/authorize](https://wandb.ai/authorize) (already exists in Robert's Modal workspace as of 2026-05-23)
   - `aws-cneuromod` (only if CNeuroMod download requires AWS creds) → keys: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

**Modal app names (spec §12, locked):**

| App | GPU | Purpose |
|---|---|---|
| `sapient-1-features-vjepa2` | 1×A100-80GB | V-JEPA 2 video features |
| `sapient-1-features-w2vbert` | 1×A100-40GB | W2V-BERT audio features |
| `sapient-1-features-llama` | 1×A100-40GB | Llama-3.2-3B text features |
| `sapient-2-features-vjepa2` | 1×A100-80GB | Same, ds004996 + ds001740 |
| `sapient-2-features-w2vbert` | 1×A100-40GB | Same |
| `sapient-2-features-llama` | 1×A100-40GB | Same |
| `sapient-1-train-llama` | 1×H100-80GB | Sapient-1 from scratch |
| `sapient-2-train-scratch-llama` | 1×H100-80GB | Sapient-2 from scratch |
| `sapient-2-train-ft-llama` | 1×H100-80GB | Sapient-2 fine-tuned from sapient-1 |
| `sapient-1-eval` | 1×A100-40GB | Vertex + parcel Pearson eval |
| `sapient-2-eval` | 1×A100-40GB | Same |

**Action item before week 1:** Run `modal app list` to verify none of these names collide with existing apps in Robert's workspace.

**Documentation links:**

- Modal docs root: https://modal.com/docs
- GPU types + pricing: https://modal.com/docs/guide/gpu
- Secrets: https://modal.com/docs/guide/secrets
- Volumes (for persistent feature caches): https://modal.com/docs/guide/volumes
- Modal pricing: https://modal.com/pricing

---

## 3. Datasets — download tooling

### CNeuroMod (4 CC0 subjects) — sapient-1 PRIMARY

**Source:** Canadian Open Neuroscience Platform (CONP)
**Method:** `datalad` clone — NOT direct S3
**Subjects to pull:** sub-01, sub-02, sub-03, sub-05 (the 4 CC0 subjects only — skip sub-04 and sub-06 which are gated)

**Install datalad:**

```bash
# macOS
brew install git-annex
pip install datalad

# Verify
datalad --version
git annex version
```

**Then clone:**

```bash
# Spec says: data/raw/cneuromod/ — confirm location with Robert
datalad install -r https://github.com/courtois-neuromod/cneuromod.git
cd cneuromod
datalad get sub-01 sub-02 sub-03 sub-05    # only the 4 CC0 subjects
```

**Documentation:**

- CNeuroMod portal: https://www.cneuromod.ca/access/access/
- CONP page: https://portal.conp.ca/dataset?id=projects/cneuromod
- datalad handbook: http://handbook.datalad.org/
- Algonauts 2025 brain data (uses same 4 CC0 subjects): https://algonautsproject.com/2025/braindata.html
- License confirmation: https://2025.ccneuro.org/abstract_pdf/Boyle_2025_CNeuroMod_Data_Collection_Complete_200h_individual.pdf

### BOLD Moments — sapient-1 AUGMENTATION (week 2)

**Source:** OpenNeuro ds005165
**Method:** `aws s3 sync` (no AWS account needed for public buckets) or datalad

```bash
aws s3 sync --no-sign-request s3://openneuro.org/ds005165 data/raw/bold_moments/
```

**Documentation:**

- OpenNeuro ds005165: https://openneuro.org/datasets/ds005165
- Paper: https://www.nature.com/articles/s41467-024-50310-3

### Narratives — sapient-1 AUGMENTATION (week 2, features-only)

**Source:** OpenNeuro ds002345
**Method:** `aws s3 sync`

```bash
aws s3 sync --no-sign-request s3://openneuro.org/ds002345 data/raw/narratives/
```

**Documentation:** https://openneuro.org/datasets/ds002345

### ds004996 NeuroEngage — sapient-2 PRIMARY

**Source:** OpenNeuro ds004996
**Method:** `aws s3 sync`

```bash
aws s3 sync --no-sign-request s3://openneuro.org/ds004996 data/raw/ds004996/
```

**Documentation:** https://openneuro.org/datasets/ds004996

### ds001740 Rauchbauer HRI — sapient-2 AUGMENTATION

**Source:** OpenNeuro ds001740, **pin to v2.1.0**
**Method:** `aws s3 sync` with version

```bash
aws s3 sync --no-sign-request s3://openneuro.org/ds001740/versions/2.1.0 data/raw/ds001740/
```

**Documentation:**

- OpenNeuro ds001740: https://openneuro.org/datasets/ds001740/versions/2.1.0
- Paper: https://pubmed.ncbi.nlm.nih.gov/30852994/

---

## 4. Weights & Biases (training logs)

**Account:** confirm Robert has a W&B account; if not, create at [wandb.ai](https://wandb.ai).

**Project to create:** `sapient` (under Robert's W&B user or team).

**Credentials Claude needs:**

1. `WANDB_API_KEY` — from [wandb.ai/authorize](https://wandb.ai/authorize)
2. Set as env var: `export WANDB_API_KEY=...`
3. Also add to Modal secrets (see §2).

**Documentation:**

- Quickstart: https://docs.wandb.ai/quickstart
- PyTorch integration: https://docs.wandb.ai/guides/integrations/pytorch
- Logging best practices: https://docs.wandb.ai/guides/track

---

## 5. Encoder model downloads

These are downloaded once and cached. Verify each works before week 1.

| Model | HF repo | License | Gated? |
|---|---|---|---|
| V-JEPA 2 Gigantic | `facebook/vjepa2-vitg-fpc64-256` | MIT | No |
| W2V-BERT 2.0 | `facebook/w2v-bert-2.0` | MIT | No |
| Llama-3.2-3B | `meta-llama/Llama-3.2-3B` | Llama 3.2 Community | **Yes — already approved** |
| Whisper-large-v3 (ablation) | `openai/whisper-large-v3` | MIT | No |

Smoke test (run before any feature extraction):

```python
from transformers import AutoModel, AutoTokenizer
for repo in ["facebook/vjepa2-vitg-fpc64-256",
             "facebook/w2v-bert-2.0",
             "meta-llama/Llama-3.2-3B"]:
    print(f"Testing {repo}...")
    AutoModel.from_pretrained(repo, torch_dtype="auto")   # downloads + loads
    print(f"  ✅ {repo}")
```

---

## 6. Local dev environment

**Python:** 3.11 (matches Modal default)
**Package manager:** `uv` recommended — https://docs.astral.sh/uv/

Core packages (will be pinned in `pyproject.toml`):

- torch ≥ 2.4, torchaudio, torchvision
- transformers ≥ 4.45
- huggingface-hub ≥ 0.25
- modal ≥ 0.66
- wandb, nilearn, nibabel, datalad, pyyaml
- pytest (for the sanity tests in spec §3.4)

---

## 7. Day-0 readiness checklist (Claude runs this with Robert)

Before any repo scaffolding, confirm each:

- [ ] `hf auth whoami` → returns Robert's username
- [ ] `hf download meta-llama/Llama-3.2-3B --include "config.json"` → succeeds
- [ ] `huggingface.co/The-Sapient-Company` → page exists, Robert is admin
- [ ] HF write token created, named `sapient-build-token`, scoped to the org
- [ ] `modal config show` → returns valid token (note: `modal token current` referenced in older docs does not exist in current Modal CLI; auth confirmation comes from any successful `modal app list` / `modal secret list` call)
- [ ] `modal app list` → none of the 11 spec §12 app names exist yet
- [ ] HF + W&B secrets added to Modal dashboard
- [ ] `datalad --version` and `git annex version` → both work
- [ ] `aws --version` → installed (no creds needed for OpenNeuro public buckets)
- [ ] `wandb login` → succeeds, `sapient` project exists or is creatable
- [ ] Smoke test in §5 above → all 3 encoders load locally

Only when every checkbox is ✅ do we scaffold `sapient1/` and write code.

---

## 8. Patch instruction for Claude on existing specs

Anywhere the existing engineering-outline docs say:

- `sapient/sapient-1-llama` → read as `The-Sapient-Company/sapient-1-llama`
- `sapient/sapient-2-scratch-llama` → `The-Sapient-Company/sapient-2-scratch-llama`
- `sapient/sapient-2-ft-llama` → `The-Sapient-Company/sapient-2-ft-llama`
- `sapient/{cneuromod,ds004996,ds001740}-...` → `The-Sapient-Company/{cneuromod,ds004996,ds001740}-...`
- `huggingface.co/sapient` → `huggingface.co/The-Sapient-Company`
- `HF_ORG = "sapient"` → `HF_ORG = "The-Sapient-Company"`

The literal Modal app names (`sapient-1-train-llama`, etc.) do **not** change — those are Modal app identifiers, not HF org references.

The literal Python package names (`sapient1/`, `sapient2/`) do **not** change — those are repo folder names, not HF org references.

When Claude writes code, always use `The-Sapient-Company` for any HF API call.

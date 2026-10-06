# WORKFLOW — How research becomes product

**The one source of truth for how a discovery on the brain/model side becomes a live feature in the product.**
Robert, Aayush, and any AI agent (Claude Code) read this first. Aayush's agent and Robert's agent both follow this
same document — that's how the two sides stay in sync. Last updated 2026-06-10.

---

## The idea in one line
The model **labels** every insight it produces. The frontend has a **library of display blocks**. Labels snap into
blocks **automatically**. So new research → a new card (and a new chat tool) in the product, on its own.

Think slide templates: a new insight drops into the right template by itself. Someone only builds a brand-new
template once in a while — and then it's reusable forever.

---

## Where everything lives (the map)
| Thing | Where | Owner |
|---|---|---|
| Models (predict the brain response) | `sapient-models/mary/`, `sapient-models/qualia/` | Aayush |
| Insight readouts (the scores/constructs) | `sapient-models/neurosignal/` | Aayush |
| **The handoff boundary** (where the two agents meet) | **`sapient-models/handoff/`** | shared |
| ↳ the contract (what the model outputs + how to show it) | `handoff/capabilities.json` | Aayush |
| ↳ real sample output the frontend renders against | `handoff/samples/<model>/` | Aayush |
| ↳ what changed each push (newest on top) | `handoff/CHANGELOG.md` | both |
| Frontend + the display blocks | the `sapient` repo | Robert |
| **The Component & API Library** (see every block + the JSON it renders) | the app at **`/ops/components`** | Robert |
| **This workflow** | `sapient-models/WORKFLOW.md` (here) | shared |

> `ag_push` is **retired.** Everything Aayush makes goes into **`sapient-models`** — nowhere else.

---

## Two agents, one source of truth (this is the whole point)
When Aayush pushes something, **his agent and Robert's agent both read THIS file + the `handoff/` folder.** That shared
boundary is the synchronization. Neither side has to understand the other's code — they meet in `handoff/`.

**`handoff/` is the drop point** (full detail in `handoff/README.md`): Aayush's agent *writes* the capability into
`handoff/capabilities.json`, drops a real sample in `handoff/samples/<model>/`, and appends a line to
`handoff/CHANGELOG.md`. Robert's agent *reads* those three to know what's new and render it. If it's not in `handoff/`,
the frontend doesn't know about it.

**Aayush's agent** (works in `sapient-models`):
1. Reads `WORKFLOW.md`.
2. Builds the capability (a `neurosignal` readout, or a model change), **adds one entry to `handoff/capabilities.json`**
   (label + display block + honest caption), and drops a real sample in `handoff/samples/<model>/`.
3. Branches `feature/<name>`, opens a **PR into `main`**. Never pushes to `main`, never deploys on merge.
4. Its job ends when the capability is **declared in the manifest, merged, and the new model is deployed + promoted.**

**Robert's agent** (works in the `sapient` app):
1. Reads `WORKFLOW.md` + the latest `capabilities.json`.
2. For each capability: if its `display` block already exists → **nothing to do, it auto-renders.** If it's a new
   *shape* → scaffold the new display block, add it to the **Component & API Library at `/ops/components`** with the
   sample JSON, then it's reusable forever.
3. Branches `feature/<name>` in the app, PR into `main`.

**The handoff artifact between the two agents = `capabilities.json` + the sample output.** That's the contract. As long
as both agents read this file, work pushed from either side stays synchronized.

---

## How we ship — research → live, end to end
1. **Aayush builds** the capability (usually a readout in `neurosignal`; sometimes a retrained model).
2. **He declares it** — one entry in `handoff/capabilities.json` + a real sample in `handoff/samples/<model>/`.
3. **PR into `main`** → CI **eval gate** runs (does it beat the current model on brain metrics?) → pass/fail on the PR.
4. **Merge** the green PR. *(Merging accepts the code — it does NOT change production.)*
5. **Deploy:** `modal deploy` the serving → **promote by flipping the tag** the live endpoint serves. *(This is the
   step that makes it live. Apps: `qualia-serve`, `mary-serve`.)*
6. **The app auto-syncs** to the live endpoint → the new capability appears in the product. If it needed a new display
   block, Robert's side ships that (steps in "Two agents" above). **Rollback = flip the tag back. Instant.**

### Merge vs deploy — the rule that protects production
- **Merge to `main`** = code accepted. Does nothing to prod by itself.
- **Deploy/promote** = the separate, deliberate step that goes live.
- **Never let a merge auto-deploy** — so a bad merge can never take `qualia-serve` down.
- **Merge:** model code · `neurosignal` readouts · `handoff/` (capabilities.json + samples).
- **Do NOT merge:** scratch/experiments (keep on experiment branches) · anything red on the eval gate.

---

## The Component & API Library (`/ops/components`)
The place to **see** the system and **test** new components before they go live. Tabs: **Mary · Qualia · All
Components**, with **per-lens** views (Attention · Emotion · Memory · Buy/Not Buy · Manipulation). For each component it
shows the live render with sample data **and the raw API JSON** that drives it — so the team can see exactly what each
model + lens outputs. When we build or change a component, we run it through here first (sample data → check the JSON →
keep notes), then ship it. **Every model update should refresh the sample JSON here**, so this doubles as the living
API-response reference.

---

## The one rule
- **Reuse (almost always):** the new insight uses a display block that already exists → it appears automatically,
  nobody builds anything.
- **Build new (rarely):** the insight is a new *shape* → build ONE block, register it at `/ops/components` → it's in
  the library forever after.

**Aayush makes the model *say* new things. Robert builds the *ways to show things* once. The manifest matches them. Both
agents read this file.**

---

## For an AI agent working either repo
- Read this file first. If you're in `sapient-models`, also read `README.md`; if you're in the app, also read
  `docs/components-by-lens.md` + look at `/ops/components`.
- Model/insight work → `sapient-models` only (`neurosignal/` + an entry in `handoff/capabilities.json` + a sample in `handoff/samples/`). Frontend → the `sapient` repo.
- Always work on a `feature/<name>` branch → PR into `main`. **Never push to `main` directly. Never deploy on merge.**
- Preserve infra names exactly: Modal apps `qualia-serve` / `mary-serve` / `kairo-extract`, all `kairo-*` volumes,
  HF org `The-Sapient-Company`, every model release name ends in `-llama`, Llama-3.2-3B is the only text encoder.
- Every capability MUST carry `covered` + `honesty` — never surface a metric the input can't actually measure.

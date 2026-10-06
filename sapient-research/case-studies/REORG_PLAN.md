# Repo organization & naming — cleanup plan

Goal: make the architecture easier to read without colliding with the teammate's active work
(PR #96) or breaking imports across `server.ts` (161 KB) and 69 `src/` dirs.

## Hard constraints (from `CLAUDE.md` and repo state — do not violate)
- **Do not rename `Kairo`** — intentional persona identity in `src/sapientPersona.ts`.
- **Do not split large component files** unless explicitly asked.
- **Do not reintroduce `Simplr`** — old name, rebrand complete.
- **`ag_push/` is gitignored** — a nested clone of the *separate* repo
  `github.com/The-Sapient-Company/ag_push` (350 MB model data). **Renaming it here does
  nothing to the real thing and desyncs the mirror.** Rename must happen in that repo.

## The actual naming confusion (what makes this hard to read)
| Name | Where | Problem |
|---|---|---|
| `ag_push` | gitignored nested repo | Personal-initials name for what is really the **prediction engine**. Reads as noise. |
| `prediction-suite` | inside `ag_push/` | Good name, but buried under `ag_push`. |
| `neurosignal` | top-level Python pkg | Fine, but its relationship to `prediction-suite` (they overlap: both do readouts) is undocumented. |
| `Mary`, `TRIBE v2` | encoder, on Modal | Model codenames scattered across docs with no single glossary. |
| `evals/`, `sapienteval/`, `sapient-serving/` | three top-level dirs | Three eval/serve homes; unclear which is canonical. |
| `verdict-v3`, `sapient-scan` | `src/` | Product feature names; fine, but no index maps feature → dir. |

## Recommended target (proposal — needs owner + teammate sign-off before code moves)
1. **Rename the external `ag_push` repo → `prediction-engine`** (do it in that repo; update the
   one comment in `src/lib/neuroInterpret.ts:5` and `.gitignore` here to match).
2. **Add a top-level `GLOSSARY.md`**: one table mapping every codename → what it is
   (`Mary` = fMRI encoder service, `TRIBE v2` = encoder weights, `neurosignal` = construct
   engine, `prediction-engine` = behavioral bridge, `Kairo` = persona voice). Zero-risk, biggest
   readability win. **← do this first.**
3. **Consolidate eval dirs:** document in the glossary which of `evals/ | sapienteval/ |
   sapient-serving/` is canonical; fold the dead one into the live one in a *dedicated* PR
   (touches Python imports — isolate it).
4. **`src/` feature index:** add a short "feature → directory" table to `CLAUDE.md`
   (`verdict-v3`, `sapient-scan`, `landing-experiments`, `admin`). Doc-only, zero-risk.

## Executed now (zero-risk only)
- ✅ `case-studies/` — new, self-contained, imported by nothing, excluded from deploy.
- ✅ `.vercelignore` — keeps internal material out of customer builds.
- ✅ This plan + `GLOSSARY.md` stub (doc-only).

## Deferred (needs sign-off — NOT executed)
- Any rename touching `server.ts` / `src/` imports.
- The `ag_push` → `prediction-engine` rename (belongs in the other repo).
- Eval-dir consolidation (own PR, run the Python tests after).

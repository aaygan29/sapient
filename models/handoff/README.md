# handoff/ — the boundary between the model side and the frontend

**This folder is where Aayush's agent and Robert's agent meet.** It's the one place the two sides communicate.
Read `../WORKFLOW.md` first for the full picture; this is the drop point that workflow refers to.

## How it works
- **Aayush's agent WRITES here** when a model/insight changes:
  1. Adds/updates the capability in `capabilities.json` (the contract — what the model can now output + how it should
     be shown).
  2. Drops a real sample output in `samples/<model>/` (the JSON the frontend renders against — never mocked).
  3. Appends one line to `CHANGELOG.md` ("what's new this push").
- **Robert's agent READS here** to build/verify the frontend:
  1. Reads `CHANGELOG.md` (top = newest) to see what changed.
  2. Reads `capabilities.json` — if a capability's `display` block already exists, it auto-renders; if it's a new
     *shape*, scaffold a new display block and register it in the app's `/ops/components` library.
  3. Renders against `samples/<model>/` so the UI matches the real output exactly.

## Files
| File | What it is | Who writes |
|---|---|---|
| `capabilities.json` | The contract: every insight the models output + display block + honesty flags | Aayush's agent |
| `samples/<model>/` | Real sample output JSON the frontend renders against | Aayush's agent |
| `CHANGELOG.md` | Running log of what changed each push (newest on top) | Both (Aayush adds; Robert notes what he built) |

## The rule
Everything that crosses from model → frontend goes through this folder. If it's not in `capabilities.json` with a
sample, the frontend doesn't know about it. Every capability MUST carry `covered` + `honesty` — never surface a metric
the input can't actually measure.

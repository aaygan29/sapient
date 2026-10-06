# Sapient Agents — Canvas Workflow Builder (build spec)

_Last updated 2026-08-12. Source: Robert's canvas vision (screenshots + voice notes)._

## The goal, in one paragraph
A node canvas that autonomously optimizes an ad video toward an outcome. You drop
in a video; the Sapient model decodes it (brain + API response); you state a goal
in plain language ("I want more conversions"); an agent — with **full understanding
of the video (transcript, visuals, audio/tone, brain response)** — diagnoses exactly
which signals to change, proposes the edits, and **on your approval** executes them:
cuts the video, pulls the target frame, writes its own generation prompt, generates
the new footage/audio via AI models, and assembles **multiple optimized ad variants**
— each targeting a specific brain response and **pre-scored with a predicted brain
response** so real ad analytics can be compared against it later. All live, in the
canvas, watching the agent work.

## Node graph (left → right)
1. **Video** — upload a real video (player + duration). No presets/dropdowns.
2. **Sapient model · API response** — connected to the video. Brain render stays
   visible; below it the **API response**, **searchable**, each signal **color-coded**
   — 🟢 okay (leave it) · 🟠 recommended change · 🔴 optimize — with a **legend key**
   and the API key shown. This is the raw decode the agent reasons over.
3. **Understanding** _(new)_ — transcript (STT), visual scene breakdown (frames →
   vision model), audio/tone. Gives the agent full context to make the right calls.
4. **Goal** — free text the user types (their outcome). No preset dropdown, no fake
   "strength" metric.
5. **Agent** — reads API response + understanding + goal → diagnoses the specific
   signals to change and writes the change plan ("cut 0:08–0:10; add a product-in-hand
   shot at 0:09 to lift reward; rewrite the CTA line; warmer VO tone"). **User approves.**
6. **On approve → the agent spawns execution nodes**, one branch per change:
   - **Cut** (ffmpeg) — trim/segment at the timestamps.
   - **Frame** (ffmpeg) — extract the target frame (e.g. last frame of the section).
   - **Prompt** (Claude) — the model writes the generation prompt for that section:
     what fills it (another 5s, a spoken line, a physical element), matched to the goal.
   - **Image** (fal.ai Flux) — generate/edit the still.
   - **Video** (fal.ai Kling/Seedance) — turn the still into the clip.
   - **Voice** (ElevenLabs) — new VO line / tone.
   - **Assemble** (ffmpeg) — splice everything into the cut video.
7. **Variants + predicted score** — the agent produces **multiple variants**, each
   optimized for a specific brain response. Each finished variant is **re-run through
   the model → predicted brain response**, shown beside it, downloadable, and stored so
   future real analytics can be measured against the prediction.

## Everything the canvas consumes (tool / model inventory)
| Capability | Tool / model | Key | Status |
|---|---|---|---|
| Brain decode (API response) | Sapient scan API (`/run_scan`) | `QUALIA_SERVE_TRIGGER_URL` | trigger DOWN → pre-scored until restored (owner) |
| Agent reasoning (diagnose, plan, prompt-gen) | Claude (`api/chat.ts`, Sonnet) | Anthropic | **live** |
| Transcript / STT | Whisper (fal or local) | `FAL_KEY` / local | to wire |
| Visual understanding | Claude vision on frames | Anthropic | to wire |
| Cut / frame / splice / render | ffmpeg (local) | — | **live (free)** |
| Image gen / edit | fal.ai Flux | `FAL_KEY` | key valid, **$0 balance → top up** |
| Image → video | fal.ai Kling / Seedance | `FAL_KEY` | same |
| Voice / tone | ElevenLabs | `ELEVEN_API_KEY` | **live (verified, 25 voices)** |
| Predicted re-score | Sapient model on the variant | scan trigger | ties to trigger restore |
| Storage (source + assets) | Supabase Storage | (have) | **live** |

## Live "computer usage" layer
Every node carries an execution state (idle → running → done → error) with a live
status line, streamed from an SSE orchestrator route (`api/agent-run`). Nodes appear
**progressively** as the agent decides steps; intermediate artifacts render inline as
they're produced (extracted frame → generated still → clip → final variant); edges
animate when data flows. You watch the agent operate the tools.

## Build phases
**Phase 1 — the spine, all free (buildable now):**
Real video upload → Supabase; Understanding node (STT + frames + tone); API-response
node on real decode (pre-scored until trigger back), searchable + color-coded (UI done,
wire real data); Agent diagnose → change plan → **approve gate**; execution nodes run
the free steps end to end (cut, frame, prompt-gen, VO, assemble w/o the AI insert);
live SSE execution (states + logs + inline artifacts + progressive build); variants
scaffold + predicted-score slots.

**Phase 2 — generative insert (unblocks on fal.ai balance):**
Image + Video nodes fire for real; generated footage splices into the variants; final
MP4s render with the new visuals.

**Phase 3 — real brain (unblocks on owner action):**
Restore `QUALIA_SERVE_TRIGGER_URL` + redeploy → real brain read on the uploaded video
and real predicted re-score on each variant.

## Blocked on Robert (the only external gates)
1. **fal.ai balance** — top up at fal.ai/dashboard/billing (image + video gen).
2. **Scan trigger** — restore `QUALIA_SERVE_TRIGGER_URL` + redeploy (owner) for real
   decode + real predicted re-score.

_ElevenLabs, ffmpeg, Claude and Supabase are all live now — Phase 1 needs neither gate._

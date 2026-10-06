# Supabase Storage Emergency — Cleanup Plan

> Project `tndmiyfklhhpmcfxlfxl`. Storage at **216.9 GB / 100 GB (217%)**, grace period ended 2026-06-12 → requests will start 402'ing. Audit 2026-06-12 (read-only). **Nothing deleted yet — awaiting Robert's go.** Constraint: no data loss.

## The surprising headline
- The 216 GB is **~99.9% video files in storage buckets, ~0% database.** The entire Postgres DB is **197 MB** — a non-issue.
- **The duplication theory was wrong.** `kairo-uploads` and `sapient-uploads` are **disjoint generations** (≈0 shared files), not copies. Cross-bucket dedup frees ~nothing.
- The fix is pruning **video files**, not merging tables.

## Where the 216 GB lives
| Location | Size | Objects | Status |
|---|---|---|---|
| `kairo-uploads` | 79 GB | 9,134 | **LIVE** (written through 2026-06-13) — the active bucket despite the legacy name |
| `sapient-uploads` | 45 GB | 4,432 | **FROZEN legacy** (nothing since rebrand 2026-05-20); only a read-fallback |
| `sapient-research` | 161 MB | 62 | small |
| `career-resumes` | 9.5 MB | 51 | small |
| **Entire Postgres DB** | **197 MB** | — | largest table `sapient_results` = 64 MB; ignore for quota |

By type: ~99 GB mp4 + ~13 GB mov + ~5 GB wav + ~4.5 GB result JSON. By prefix: `uploads/` raw sources = **74 GB**, `videos/` playable copies = 37 GB, sidecars (`audio/`+`results/`+`mary/`) ≈ 12.5 GB.

## The 3-phase plan (216 GB → well under 100 GB)

### Phase 1 — Safe, immediate, no code change → recovers ~30 GB
1. Delete **orphaned sidecars** (no DB row references them): `audio/` (5.2 GB) + `results/` (4.5 GB) + `mary/` (2.8 GB) + `qualia/` (0.45 GB) ≈ **12.5 GB**.
2. Delete `videos/` + `uploads/` for all **`error`/`cancelled` jobs** (1,119 jobs, never produced a result) ≈ **8–12 GB**.
3. Delete **truly-orphan** source videos (no job at all) ≈ **10 GB**.
→ gets you ~217% → ~155 %. Breathing room, zero risk.

### Phase 2 — The big lever, ONE product decision → recovers ~50–60 GB (this solves it)
4. For **`complete` jobs**, delete the **raw `uploads/` source video** once the normalized `videos/<id>.mp4` exists. The raw source is only needed during processing.
   - **Decision needed from Robert:** *once a scan is complete, do we ever re-process from the original upload?* If **no** → delete completed-job `uploads/` sources (keep recent <30d as a buffer) → recovers ~50–60 GB → **safely under 100 GB.**

### Phase 3 — Retire the legacy bucket, needs code change → up to 45 GB headroom
5. Remove the `sapient-uploads` fallback reads (`api/analyze.ts`, `api/share/[id].ts`, `api/account/delete.ts`, `server.ts`, `AnalyzeTab.tsx`). For the ~41 GB still referenced by old completed jobs: either copy into `kairo-uploads` (preserve playback) or accept pre-2026-05-20 jobs lose video. Then delete the bucket. **This is also exactly the Kairo bucket-rename Stage 3** — one effort, two wins.
6. Add a bucket file-size cap + a post-completion TTL that auto-deletes `uploads/` sources, so this never recurs.

## Hard safety rules before ANY delete
- Generate the concrete delete list with a **full-path anti-join** (storage object name vs `sapient_jobs.source_url` / DB path columns), NOT a basename match — ~773 `uploads/` videos look orphaned by basename but are actually referenced by full path. Deleting on basename = data loss.
- Delete by **copy-to-a-manifest-first** (write the list of keys to be deleted to a file) so every deletion is auditable and a recovery list exists.
- Phase 1 deletions are irreversible — get explicit go.

## Recommendation
Approve **Phase 1 now** (zero-risk ~30 GB). Answer the **Phase 2 question** — if "we don't re-process from raw," Phase 2 ends the emergency by itself. Do **Phase 3 with the Kairo bucket migration** as one coordinated effort.

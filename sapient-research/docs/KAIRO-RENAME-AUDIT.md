# Kairo → Qualia/Sapient Rename — Full Break-Point Map & Staged Runbook

> Detailed dependency audit, 2026-06-12 (both repos). **Nothing renamed yet.** The audit proves a blind one-shot rename crashes the live serving engine and breaks 50+ DB queries. This is the safe, staged, no-data-loss path. Constraint from Robert: **we cannot lose any data.**

## The 7 live systems "kairo" touches (every one is a break-point)

| # | System | Where | If renamed wrong |
|---|---|---|---|
| 1 | **`kairo-uploads`** Supabase bucket (79 GB / 9,134 obj) | app: ~38 refs (server.ts, analyze.ts, mary-run.ts, sapient-scan.ts…); models: qualia+mary serve.py both write here | user videos/scans 404; **shared by BOTH qualia-serve & mary-serve** |
| 2 | **Modal volumes** `kairo-weights` / `kairo-features` / `kairo-extracted` / `kairo-hf-cache` | models: qualia/modal/serve.py:68-72, qualia/serving/extract.py:75,80 | **live engine crashes at startup** — checkpoint `/weights/v4_seed_0/best_model.pt` unreachable |
| 3 | **Modal app `kairo-extract`** | models: extract.py:58; serve.py:277,334,626 (3 cross-app calls) | qualia-serve `Function.from_name` fails immediately |
| 4 | **DB views** `kairo_jobs`, `kairo_results`, `kairo_comparisons`, `kairo_health_checks` | app: 50+ query sites across server.ts, analyze.ts, admin/* | every scan read/write query errors |
| 5 | **Vercel env vars** `KAIRO_PIPELINE_SECRET`, `KAIRO_ADMIN_PASSWORD`, `KAIRO_SERVE_DIR` | app: analyze.ts, mary-run.ts, sapient-scan.ts, App.tsx, vite.config.ts | pipeline auth + admin login break until Vercel is updated |
| 6 | **localStorage keys** `kairo_admin`, `kairo_dev_role`, `kairo_dark_mode`, `kairo_session_id`… (~9) | app: App.tsx, AdminConsole.tsx, AnalyzeTab.tsx | users logged out / lose local state on deploy |
| 7 | **Modal trigger URLs** `kairo-analyze-cpu-*`, `kairo-intelligence-pipeline-*` | app: analyze.ts:859-860, admin/pipeline.ts | document/pipeline triggers 404 |

## Safe to rename NOW (code identifiers — `tsc`/imports catch any miss; zero live-resource impact)
- `KairoArtifact` type → `QualiaArtifact` (module-private to `api/_qualiaRun.ts`, 4 uses)
- `"kairo-v4"` model metadata string → `"qualia-v4"` (models serve.py:183,211 — cosmetic display name)
- `load_kairo()` / `KairoEncoder` (models) — internal Python names (rename + their import sites)
- Comments, `KAIRO-###` issue anchors, docstrings
- **NOT** the persona constants (`KAIRO_IDENTITY`…) — that's the intentional "Kairo" voice (CLAUDE.md). Separate product decision.

## My engineering recommendation (what's worth the risk)
- **KEEP the internal Modal names** (#2, #3) — `kairo-weights`/`kairo-extract` etc. are invisible to users; renaming them risks the live engine for **zero** user benefit. Your own README already decided this. Renaming them is the highest-risk, lowest-value part.
- **Rename what's user-visible or aids cleanliness** via staged migration: the bucket (#1), DB views (#4), env vars (#5), localStorage (#6), code identifiers.
- This gets you a genuinely clean codebase + the "kairo" name out of everything users/operators touch, without betting the live GPU engine on an unattended migration.

---

## The staged runbook (copy → verify → cut over → delete; each stage reversible)

**Order matters — dependencies first. Each stage is a separate reviewed deploy. Nothing is deleted until its replacement is verified live.**

### Stage 0 — Code-only safe renames (do now, `npm run lint` verifies)
Rename `KairoArtifact`→`QualiaArtifact`, `"kairo-v4"`→`"qualia-v4"`, comments. No live resource touched. Reversible via the safety snapshot.

### Stage 1 — Env vars (I can do this: VERCEL_TOKEN is in file.env)
Add `SAPIENT_PIPELINE_SECRET` etc. as **aliases** (same value) in Vercel; update code to read new-name-first, old-name-fallback; deploy; verify; remove old later.

### Stage 2 — DB views (no data move — views only)
The `kairo_*` are **views**; underlying tables are already `sapient_*`. Create `sapient_*` views (or point code at the base tables), update the 50+ query sites, deploy, verify, drop the `kairo_*` views last. **No row data is touched.**

### Stage 3 — Storage bucket (the 79 GB — the careful one)
1. New writes → `sapient-uploads` (flip `UPLOAD_BUCKET`); old reads use the **existing dual-read fallback**. Deploy + verify a fresh upload.
2. **COPY** (server-side, never move) all 9,134 objects `kairo-uploads`→`sapient-uploads`; verify counts match. *(Old data stays 100% intact = no loss possible.)*
3. Rewrite stored `kairo-uploads` URLs in `mary_runs`/jobs (tested migration + backup).
4. Soak; confirm zero fallback hits in logs.
5. Remove `kairo-uploads` from code; delete the bucket **last**.

### Stage 4 — localStorage (user state)
Rename keys with a one-time migration shim (read old → write new → delete old) so no user is logged out.

### Stage 5 (optional, NOT recommended) — Modal volumes/app
Only if you insist: create `qualia-*` volumes, copy the checkpoint + features, update serve/extract, redeploy both apps in lockstep, verify a real scan end-to-end, then delete old volumes. **This is the part that can take serving down — must be supervised with a tested rollback.**

---

## Why this isn't a "done tonight, go test" job
Stages 3 and 5 involve copying 79 GB + GPU checkpoints and redeploying the live engine, with verification that can only happen against production. Guaranteeing **"no data loss"** *requires* the copy-verify-before-delete discipline above — which is inherently multi-step and supervised, not a single autonomous pass. I'll execute it stage by stage and report verification at each.

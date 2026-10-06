# SAPIENT — CODEBASE MAP

The operating document for this repo. Read this first; everything else in `docs/` is an annex.
Verified against `origin/main` 2026-07-28. Annexes: `docs/ARCHITECTURE.md` (NeuroRead data spine),
`docs/USAGE-BILLING-SPEC.md`, `docs/KAIRO-RENAME-AUDIT.md`, `docs/SUPABASE-CLEANUP-PLAN.md`,
`src/verdict-v3/ARCHITECTURE.md` (per-component wiring).

**Stack:** React 19 + Vite 6 + TypeScript SPA · Vercel serverless `api/` · Supabase (Postgres +
storage) · Clerk auth · Stripe billing · Modal GPU workers · Express dev server (`server/index.ts`).
`"type": "module"` — relative TS imports in `api/` carry `.js` extensions.

---

## 1. Entry points & routing

### `src/main.tsx` — pathname gates (rendered OUTSIDE ClerkProvider)

| Path | Component | Gate |
|---|---|---|
| any (maintenance) | inline `MaintenancePage` | `VITE_MAINTENANCE_MODE === 'true'` |
| `/share/:slug` | `src/PublicSharePage.tsx` | none — slug is the credential |
| `/case-study/:slug` | `src/CaseStudyPage.tsx` | none |
| `/test-verdict` | `src/test-verdict/TestVerdict.tsx` | **DEV only** |
| `/sapient-scan` | `src/sapient-scan/SapientScan.tsx` | **DEV only** |
| `/paywall-preview` | `src/components/paywall/PaywallPreview.tsx` | **DEV only** |
| `/virality-preview` | `src/landing-experiments/SapientViralityLanding.tsx` | none (live) |
| `/labs`, `/labs/*` | `src/labs/*` | none |
| `/brain` | `src/brain/BrainPage.tsx` | none |
| everything else | `<ClerkProvider>` → `<PostHogIdentify/>` + `<App/>` | — |

Query-param preview overrides captured to sessionStorage before App boots:
`?tour=` → `sapient:forceTour`, `?onboarding=` → `sapient:onboardingVariant`, `?view=` → `sapient:viewOverride`.

### `src/App.tsx` — the SPA monolith (~780 KB)
`View` union at ~line 99 (58 members). `getInitialView()` (~404–626) maps URL → view;
reverse URL sync ~1233–1272. Key routes:

| URL | view | Component |
|---|---|---|
| `/` `/home` | `home` | `src/landing/LandingHome.tsx` (localhost only — prod serves the static homepage, see §6) |
| `/signin` `/login` `/signup` | `signin` | `src/AuthPage.tsx` (bonus-code gated signup) |
| `/analysis` (+ legacy `/beta*`, `/analyze`) | `analyze` | `src/AnalyzeTab.tsx` (575 KB) |
| `/intelligence` `/mary` | `mary-chat` | `src/MaryWorkspace.tsx` (lazy, full-screen) — **THE product** |
| `/library` | `scan-library` | `src/ScanLibrary.tsx` |
| `/settings`, `/api` (signed-in) | `settings` | `src/BillingPanel.tsx` + `src/ApiKeysSection.tsx` |
| `/intel(/:slug)` `/research(/:slug)` | intel/research | inline + `src/research/*` |
| `/careers(/:slug)` | careers | `src/CareersTab.tsx` / `src/CareersRolePage.tsx` |
| `/api-docs` (+`/agents`,`/reference`) | api-docs | `src/ApiDocsPage.tsx` (3 variants, outside chrome) |
| `/ops` | `ops` | `src/AdminConsole.tsx` — gated `OPS_AUTHORIZED_EMAILS` (~App.tsx:2161) |
| `/ops/components` | `ops-components` | `src/ops/ComponentLibrary.tsx` (layout/flag editor) |
| `/components`, `/manip-preview`, `/compare`, `/overview` | — | **localhost only** |
| `/live` | `live` | `src/SapientLive.tsx` — allowlist `SAPIENT_LIVE_EMAILS` |
| `/demo/<key>` | `demo` | `src/DemoAccess.tsx` (investor secret link) |
| `/individuation`, `/ripple` | — | `src/mary/IndividuationDemo.tsx` / `RippleView.tsx` |
| `/sapienteval` | `sapienteval` | `src/sapienteval/CognitiveEvalDemo.tsx` (allowlist) |
| `/paywall`, `/offer` | — | RETIRED → `/home` |

**Signed-in bounce rule (~App.tsx:1030):** `LEGACY_APP_VIEWS` redirect signed-in users to the chat.
Public marketing views (research/intel) must NEVER be in that list (bug fixed 2026-07-28).

---

## 2. Module map (`src/`)

- **`mary/`** — the 3-column chat workspace consumed by `MaryWorkspace.tsx`: `MaryConversation.tsx`
  (108 KB message stream), `MaryArtifactPanel.tsx` (right-column results), `MarySessionList.tsx`,
  onboarding variants, `RunConfirmCard`, `dashboardRegistry.tsx`, `lenses.ts`.
- **`verdict-v3/`** — the result-component library. Registry trio: `resultComponents.ts`
  (RESULT_COMPONENTS + flags), `resultSections.ts` (slots a–e, DEFAULT_LAYOUT), `resultBlocks.ts`
  (visibility). Live components: `LiveResultsEnrichment.tsx` (the modern RESULT panel: hero video +
  reactive brain, nine breakdown signals, per-second sections), `NeuroInsights.tsx`, `AttentionCurve`,
  `KpiActivationHeatmap`, `BuyMomentInsightCard`, `GroundedReport`, `BrainPanel`, `liveRichScan.ts`
  (buildLiveRichScan). Subdirs: `concepts/` (decision cards), `lensHeroes/`, `manipulation/`.
- **Brain layer** (top-level + assets): `BrainRenderer.tsx` (Three.js cortical mesh; props
  activations/hemisphere/view('lateral'|'medial'|'dorsal')/subtle), `brainRegions.ts` (Glasser →
  plain English), `public/brain/{brain_mesh,roi_map}.json`, `lib/brainColormap.ts`,
  `lib/networksToActivations.ts`. Server counterpart: `api/_brainGlossary.ts`.
- **`lib/`** — clusters: data spine (`neuroInterpret.ts` — **`interpretNeuro()` is the ONE
  translation door to `NeuroRead`; no component reads raw artifacts**), run clients
  (`maryWorkspaceClient.ts` — the frozen submit/poll contract), billing (`plans.ts`, `scanPacks.ts`,
  `gpuCost.ts`), auth/infra (`clerkShim.tsx` dev bypass, `posthogClient.ts`), demo data
  (`maryRuns.ts` = frozen investor numbers, NOT runtime).
- **`admin/` + `ops/`** — `/ops` console tabs (`AdminDashboard`, `AdminUsersTab`, `AdminPipelineTab`,
  `AdminApiTab`…) and the `/ops/components` editor (`ComponentLibrary.tsx`, 106 KB).
- **`landing/` + `labs/` + `brain/`** — React marketing surfaces (nav/footer shared via
  `MarketingNav.tsx`/`MarketingFooter.tsx`, both take `dark`).
- **Loose top-level files by area** — analyze/verdict: `AnalyzeTab.tsx`, `AnalysisV2Preview.tsx`
  (exports `VerdictLayout` — the classic results panel), `ScanLibrary`; auth: `AuthPage.tsx`;
  billing: `BillingPanel`, `ApiKeysSection`; legacy brand/creator: `Brand*`, `Creator*`,
  `CampaignManager`, `VideoReviewQueue` (retired product, localhost-gated routes); docs:
  `ApiDocsPage`; investor: `SapientLive`, `MaryWorkspace`; dev-server-only: `browserService.ts`,
  `profileExtractor.ts`, `sapientPersona.ts` (imported by `server/index.ts`, not deployed).

---

## 3. API layer (`api/`)

`api/_*.ts` = shared helpers, not routes. Key helpers: `_modelManifest.ts` (models/capabilities/
Modal trigger URLs — single source of truth), `_apiResult.ts` + `_scanShape.ts` (public shapes),
`_memory.ts` (Supabase `sapient_user_scans` + Pinecone), `_grounded.ts` (double-Claude
anti-hallucination), `_evidence.ts` (Whisper/keyframes/OCR), `_brainGlossary.ts`, `_benchmark.ts`,
`_apiLog.ts` (`withApiLogging`), `_sapient_helpers.ts` (bearer + Clerk + `CLERK_BYPASS`),
`admin/_guard.ts` (verified `__session` cookie).

**Four scan pipelines coexist:**
| Pipeline | Endpoint | Table | Modal app |
|---|---|---|---|
| Legacy v1 | `api/analyze.ts` (3.1k lines, actions submit/status/results/checkout/…) | `kairo_jobs`/`kairo_results` | `kairo-analyze-cpu-trigger-*` (primary→backup workspace failover) |
| Sapient v2 | `api/sapient-scan.ts` + `[action].ts` | `sapient_runs` | `sapient-scan-trigger` |
| Mary | `api/mary-run.ts` | `mary_runs` | `mary-serve-submit` |
| **Qualia (current)** | `api/qualia-run.ts` | `qualia_runs` | `qualia-serve-submit` |

**Qualia run trace:** `submitMaryRun()` (lib/maryWorkspaceClient) → POST `/api/qualia-run?action=submit`
→ validate capability/tier → INSERT `qualia_runs` (status queued) → `consumeCredits` (INSERT
`analysis_credits`; ceiling = `credit_overrides`) → resolve input (upload_ref → signed URL) → POST
Modal trigger `{auth_token: PIPELINE_SECRET, run_id, …}` → Modal worker writes status/progress/
result directly (service role) → frontend `pollMaryRun()` GET `?action=results` until terminal →
artifact → `interpretNeuro()` → `NeuroRead` → VerdictLayout / LiveResultsEnrichment. Failure refunds
the credit (`failRun` deletes the analysis_credits row). Historical Qualia rows also live in
`mary_runs` with `model='qualia'` — read surfaces UNION both tables.

**Other groups:** public `/v1/*` (rewritten by vercel.json; bearer `sk_live_…` → org via
`memberships`): analyze, scans, intelligence, wallet, demo (public), case-study, parcellation.
Stripe: `checkout`, `billing-portal` (dual keys: primary + `STRIPE_SECRET_KEY_LEGACY`),
`sapient/stripe-webhook.ts` (the ONLY webhook verifier; idempotency `sapient_processed_events`).
Email (Resend): `email/{welcome,scan-complete,resend-webhook,_templates}`. Ops: `admin/*`,
`ops/components`, `observability-push` (cron), `events`, `refresh-cache` (cron), `run-watchdog`
(cron). Chat/LLM: `chat` (Anthropic+Supermemory), `mary-workspace-chat`, `mary-content-analysis`.
Legacy ad-intelligence: `sapient-intelligence/*` + `lib/meta/*`. **No Clerk webhook handler exists.**

---

## 4. Data model (Supabase `tndmiyfklhhpmcfxlfxl`)

Migrations: `supabase/migrations/` (34+, canonical). Row-count estimates in `list_tables` LIE —
always `count(*)`.

**Live tables (by reference count):** `kairo_jobs`(54)/`kairo_results`(22) — legacy-NAMED but LIVE
job tables for /api/analyze; `mary_runs`(39) + `qualia_runs`(18) — GPU run rows, artifact in
`result` jsonb; `sapient_runs`(33); `user_profiles`(35); `credit_overrides`(33) + `analysis_credits`(21)
— the scan-credit system; `sapient_deposits`(20) — the API dollar wallet (**separate money system,
never cross them** — see src/lib/scanPacks.ts header); `sapient_api_keys`(19) — prefix + bcrypt
secret_hash; `organizations`(17)/`memberships`(16) — org scoping for everything; `sapient_waitlist_4`(16)
— THE live waitlist; `email_events`; `admin_audit_log`; `user_events`; `chat_sessions/messages`;
`sapient_usage_events`; `bonus_codes`(+redemptions); `approved_accounts` (invite gate);
`sapient_request_log`; `admin_settings`; `result_layouts`/`component_flags`/`component_feedback`;
`sapient_user_scans` (org API memory → /v1/scans, /v1/intelligence, mirrored to Pinecone);
`sapient_shares` (slug `shr_*`); `sapient_benchmarks`; `user_preferences`; `sapienteval_*`;
`mary_run_logs`/`mary_grounded_cache`; `sapient_brain_waitlist` (/brain page).

**Stale/legacy still present:** `sapient_jobs` (4,999 rows, dead since 5/14 — only api/share/create
references it), `platform_*` + `sync_jobs` (retired Meta/Shopify ingestion), `raw_feed_items`/
`cached_articles`/`user_signals` (old intel pipeline), `kairo_health_checks`, `beta_signups`,
`waitlist`/`sapient_waitlist`/`_3`/`_5` (superseded), `guide_leads`, `score_jobs`, `sapient_results`
(64 MB blob), `kairo_comparisons`.

**Storage buckets:** `kairo-uploads` **79 GB — LIVE primary** (legacy name; prefixes uploads/,
videos/, qualia/, mary/); `sapient-uploads` 45 GB frozen (read-fallback); `sapient-research`;
`career-resumes`; `investor-assets` (public read-only, created 2026-07-28).

---

## 5. External services (exact locations)

| Service | Where |
|---|---|
| Clerk | provider `main.tsx`; dev bypass `src/lib/clerkShim.tsx` via `VITE_CLERK_BYPASS` (+`VITE_CLERK_SIGNED_OUT` for anonymous mock); server `api/admin/_guard.ts`, `_sapient_helpers.ts` (`CLERK_BYPASS` + `Bearer dev-bypass-<id>`); NO webhook |
| Stripe | `api/checkout.ts`, `billing-portal.ts` (dual account), `sapient/stripe-webhook.ts`; prices env-var-driven via `priceEnvVar` in `src/lib/plans.ts`; packs `src/lib/scanPacks.ts` |
| Supabase | browser `src/supabaseClient.ts` (anon); server per-file service-role singletons |
| Modal | HTTPS POST to `https://${MODAL_WORKSPACE_PRIMARY}--<app>.modal.run` (workspace default `robert-16572`); apps: qualia-serve, mary-serve, sapient-scan-trigger, kairo-analyze-*; auth = `*_PIPELINE_SECRET`; failover in api/analyze.ts:895–990; cost `src/lib/gpuCost.ts` |
| Resend | `api/email/*`; templates `_templates.ts`; Svix-verified webhook |
| Pinecone | only `api/_memory.ts` |
| Supermemory | `api/chat.ts`, `api/profile.ts`, `api/memory/preload.ts` (container tag `kairo_${userId}` — rename-risky) |
| LLMs | Anthropic (`chat`, `_grounded`, `mary-content-analysis`), OpenAI (`score`, `signal`, `insights`, eval), Exa (`signal`) |
| PostHog / Vercel Analytics / Intercom / New Relic | `lib/posthogClient` + `PostHogIdentify`; `main.tsx`; `App.tsx` (SDK); `lib/newrelic.ts` + `observability-push` cron |
| Whisper (self-hosted) | `WHISPER_TRANSCRIBE_URL` in `api/_evidence.ts` |
| Published packages | `packages/sapient-mcp` (the `mcp__sapient__*` tools), `packages/sapient-cli` |

Env-var inventory: `.env.example` is STALE — ~60 additional names exist only in code (all
`STRIPE_*_PRICE_ID`s, `MODAL_WORKSPACE_*`, `*_PIPELINE_SECRET`, `PINECONE_*`, `RESEND_*`,
`SAPIENT_DEMO_RUN_IDS`, `CASE_STUDY_RUN_IDS`, `DEMO_ACCESS_KEY`, `EVIDENCE_*`, `RUN_WATCHDOG_MINUTES`…).
Clerk secret lives ONLY in Vercel as a sensitive var (cannot be pulled — see ops memory).

---

## 6. Build & deploy

- **vite.config.ts:** hidden sourcemaps; function-based manualChunks (vendor-react/-clerk/-three/
  -supabase); `@clerk/react` → clerkShim alias when `VITE_CLERK_BYPASS=true`; dev middleware
  serves `/` → `/qoves-clone/index.html` (prod parity); define injects only public envs.
- **vercel.json:** crons (`refresh-cache` daily, `observability-push` */10, `run-watchdog` */15);
  maxDurations (analyze 60s); rewrites order: `/api/*` → `/v1/* → /api/v1/*` → `/new/` →
  **`/` → `/qoves-clone/index.html`** → SPA catch-all (excludes api/, _vercel/, assets/, new).
  Redirects: askkairo.com → thesapientcompany.com (KEEP — old links/emails), /beta → /analysis,
  /docs → /api-docs. Security headers global; `/api/*` no-store.
- **middleware.ts (edge):** homepage rewrite (fires BEFORE filesystem routing), scanner-UA blocks
  (403), vuln-path blocks (404 on purpose), curl block, in-memory rate limit (60 api/min).
- **server/index.ts** — dev-only Express (3.4k lines); `npm run dev` = `tsx server/index.ts`; mirrors api/
  functions + dev-only routes (voice, browse, oauth, batch). NOT deployed.
- **Deploy-verify law:** the SPA catch-all 200s ANY path — verify deploys by grepping served
  content, never by asset status codes.

## 7. Conventions

1. **Adding a page:** outside auth → `IS_*` gate in main.tsx (+ render branch); inside SPA →
   View union + getInitialView + URL-sync + render arm (or early-return for full-screen).
2. **Static homepage** (`public/qoves-clone/index.html`, ~2.8 MB): append `<style id="sapient-*">`
   blocks; NEVER edit captured markup/CSS directly. Noir theme = the `sapient-noir` block
   (v1→v7 layers). Card frames in the donor design use CSS `outline`, not border.
3. **Theming:** `isDarkMode` in App (localStorage `sapient_theme_dark_v2`) + `<html class="dark">`;
   marketing pages take explicit `dark` props; `.sic-*` class family for internal components.
4. **Results stack:** artifact → `interpretNeuro()` → `NeuroRead` → VerdictLayout
   (AnalysisV2Preview) / LiveResultsEnrichment (verdict-v3) → slots a–e → components.
   The nine breakdown signals = `rich.composites` (5) + interpretNeuro metrics (Attention,
   Buy Signal) + constructs (Hesitation/Risk, Mental Effort — lower is better).
5. **Money:** scan credits (`credit_overrides` + `analysis_credits`) vs API wallet
   (`sapient_deposits`) — separate systems, never mixed. Every failed run refunds its credit.
6. **IDs:** runs `qualia_run_<ts>_<rand>`; shares `shr_xxxxxxxx`; API keys `sk_live_` + 16-char
   prefix + bcrypt'd secret.
7. **Allowlists are inline const arrays**: `OPS_AUTHORIZED_EMAILS`, `SAPIENT_LIVE_EMAILS`
   (App.tsx), `src/ops/opsAllowlist.ts`, `src/sapienteval/allowlist.ts`.
8. **Branch note:** file presence differs across old branches (e.g. `LiveResultsEnrichment.tsx`
   exists on main but not on `turing/secure-chat-credits`). Always check against origin/main.

## 8. Kairo rename backlog (live systems that still carry the old name)

Do NOT rename casually — each needs its migration. Full audit: `docs/KAIRO-RENAME-AUDIT.md`.

| System | Blast radius | Migration recipe |
|---|---|---|
| `kairo-uploads` bucket (79 GB, 9,134 objs) | 25 code sites + Modal apps write to it | server-side COPY → rewrite stored URLs → flip `UPLOAD_BUCKET` → soak on dual-read → delete old |
| `kairo_jobs`/`kairo_results`/`kairo_comparisons` views | 88 query sites | create sapient-named views → repoint code → drop old views last |
| `kairo_health_checks` table | 5 sites (ops health) | ALTER TABLE RENAME + update sites |
| `KAIRO_PIPELINE_SECRET` env | 2 sites read with NO fallback (api/analyze.ts:104, admin/pipeline.ts) | add SAPIENT_ fallback first, then flip Vercel var |
| 9 `kairo_*` local/sessionStorage keys (`kairo_admin`, `kairo_user_id`, `kairo_session_id`…) | logout/re-anonymize risk | boot shim: read-old → write-new → delete-old, keep 1 release |
| `askkairo.com` redirects + `robert@askkairo.com` allowlist entries | old links/emails + owner login | keep redirects forever; verify login identity before touching allowlists |
| Modal `kairo-*` apps/volumes, Resend template `kairo-email-scan-complete`, Stripe promo `kairo2026`, Supermemory tag `kairo_${userId}`, DB `model_version:'kairo-v2'`, `skuForModel('kairo')` | external systems + stored data | keep (recommendation of both audits) |

FALSE POSITIVE: "KAIROS." in an intel article (App.tsx ~9183) — exclude from any find/replace.
The ~120 explanatory/ticket-anchor Kairo comments are institutional memory — keep until renames land.

## Modal topology (the truth — recorded 2026-08-11 after the outage)

**The live serving workspace is `thesapientcompany`** (profile in `~/.modal.toml`):
`sapient-qualia-serve` (GPU scans; fn `submit` → `https://thesapientcompany--sapient-qualia-serve-submit.modal.run`),
`sapient-extract`, `sapient-whisper`, `sapient-mary-serve` (disabled model), `sapient-mary-api`.
Model weights live in volumes `sapient-qualia-weights` / `sapient-qualia-features` in that workspace.
All serving fns are `min_containers=0, scaledown_window=300` — scale-to-zero; first scan after idle
pays a ~1–2 min cold start.

**Trigger resolution:** `QUALIA_SERVE_TRIGGER_URL` env → else falls back to
`https://${MODAL_WORKSPACE_PRIMARY||'robert-16572'}--qualia-serve-submit.modal.run`
(`api/_modelManifest.ts`). The personal workspace `robert-16572` was EMPTIED on 2026-08-10
(cost cleanup) — anything still pointing there 404s with `modal_trigger_failed`.
`/api/v1/analyze` routes into the same qualia dispatch, so one env fixes UI + public API.

**Known-dead paths:** document/image scans (`api/analyze.ts` spike + preprocess apps) and the
legacy `sapient-scan-trigger` app point at apps that no longer exist in either workspace.
Serving code lives at `~/Desktop/Sapient-models/qualia/modal/serve.py` (app name `qualia-serve`,
env-overridable via `QUALIA_APP_NAME`).

**Law:** never delete Modal apps without grepping `api/` for `modal.run` URL constructions first;
prod points at workspaces by NAME and fails closed.

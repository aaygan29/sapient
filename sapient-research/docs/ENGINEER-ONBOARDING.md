# SAPIENT — ENGINEER ONBOARDING

The clone document. A fresh engineer (or a fresh Claude Code session) reads this and knows how the
product works, where everything lives, and which rules are load-bearing. It builds on
`docs/CODEBASE-MAP.md` (the maintained repo map, verified 2026-07-28) — read that too, but note that
the product's center of gravity has MOVED since it was written: the workspace now lives at **/app**
(`src/ds/app/`), built entirely from the design system in **`src/ds/`**, and the old `/intelligence`
Mary workspace forwards to it. Nothing in the map is wrong; this document is what changed on top.

Written 2026-08-25, verified against the working tree (which carries an uncommitted design pass —
see §8).

---

## Table of contents

1. [The 30-second map](#1-the-30-second-map)
2. [THE DESIGN SYSTEM — read this twice](#2-the-design-system)
3. [The canvas app at /app](#3-the-canvas-app-app)
4. [The scan pipeline](#4-the-scan-pipeline)
5. [The API layer](#5-the-api-layer)
6. [Operational rules you MUST know](#6-operational-rules)
7. [How to run it](#7-how-to-run-it)
8. [Current in-flight work (Aug 2026)](#8-current-in-flight-work-aug-2026)
9. [Reading list](#9-reading-list)

---

## 1. The 30-second map

**The product:** neural ad scanning. Upload a video (or paste an Instagram/TikTok/YouTube link),
a GPU pipeline on Modal produces a per-second brain read — 7 Yeo networks, KPIs, composites, a
0–100 score and a grade — and the app renders it as a live instrument: a video stage, live response
traces, a stimulus timeline and a 3D cortex, all sharing one playhead. Users iterate on a node-graph
canvas: media → scan → agent → generated variants → re-scan → comparison. Non-video inputs
(image/PDF/text) are scored by a Claude-vision estimator (`api/_neuralVision.ts`) and are always
flagged as ESTIMATES, never as measured fMRI.

**Stack:** Vite 6 + React 19 + TypeScript SPA · Vercel serverless functions in `api/` · Supabase
(Postgres + storage) · Clerk auth · Stripe (LIVE keys) · Modal GPU workers · Express dev server
(`server/index.ts`, dev-only, port 3000). `"type": "module"` — relative TS imports inside `api/`
carry `.js` extensions.

**Deploy:** pushing to `main` auto-deploys production at **thesapientcompany.com** (Vercel project
"sapient", team `the-sapient-company`). There is no staging. See §6 before you touch git.

**Top-level layout (what matters):**

```
api/                 Vercel serverless functions (each file = one endpoint; _*.ts = shared helpers)
src/main.tsx         THE entry point — a ladder of pathname gates (no React Router anywhere)
src/App.tsx          the legacy SPA monolith (marketing, /ops, old workspace) — still live
src/ds/              THE DESIGN SYSTEM + the /design style guide (§2)
src/ds/app/          the product: AppPrototype at /app (§3)
src/ds/canvas/       the node-graph primitives (canvas, nodes, wires, shells)
src/ds/brain/        TransparentBrain — the WebGL cortex
src/agents/          canvasUpload.ts (tus uploads + URL re-signing) + the older /agents board code
src/lib/             plans.ts (pricing), entitlements, clerkShim, neuroInterpret, metaPixel …
src/verdict-v3/      the legacy result-component library (LiveResultsEnrichment etc.)
server/index.ts      dev-only Express (mirrors api/ + Vite middleware). NOT deployed.
public/home/         the static marketing homepage (served at / via vercel.json rewrite)
supabase/            migrations (canonical)
docs/                CODEBASE-MAP.md, ARCHITECTURE.md, KAIRO-RENAME-AUDIT.md, this file
```

**How the app boots (`src/main.tsx`):** a chain of regex checks on `window.location.pathname`
decides which root component renders, most of them OUTSIDE the main `<App/>`:

- `/design` → `src/ds/DesignPage.tsx` — the style guide (no Clerk, no API). `main.tsx:172`.
- `/app`, `/app/signin`, `/app/signup` → `src/ds/app/AppPrototype.tsx` inside its own
  `<ClerkProvider>` (`main.tsx:175` and `262–275`). **This is the product now.**
- `/intelligence` → module-scope `window.location.replace('/app')` (`main.tsx:168`) — the redirect
  happens BEFORE App mounts because App strips `?upgraded=1` on render; old Stripe return URLs and
  bookmarks still point here.
- `/signin` and `/signup` → forwarded to `/app/signin` / `/app/signup`, stashing `?tier=` into
  `sessionStorage['sapient:intendedTier']` first (`main.tsx:156–166`) — the marketing pricing
  buttons pass the plan the visitor clicked and the paywall reads it back.
- `/library`, `/library/<id>` → `src/library/LibraryPage.tsx` / `LibraryDetail.tsx` — the public
  neural ad library, fully outside Clerk.
- `/studio` → `src/studio/StudioShell.tsx` (Neural Studio, generation surface; `/agents` redirects
  here). `/labs` → labs hub. `/neuroagi` (formerly `/brain`) → `src/brain/BrainPage.tsx`.
- `/share/:slug`, `/investors|dataroom|deck/:token`, `/virality-preview` → tokenized public pages,
  no Clerk.
- Everything else → `<ClerkProvider>` → `src/App.tsx` (the legacy monolith: marketing routes,
  `/ops`, `/analysis`, careers, intel, etc. — see CODEBASE-MAP §1 for its full route table).

`/` in production serves the static `public/home/index.html` via the vercel.json rewrite
(`"/": "/home/index.html"`) — NOT the React app. The SPA catch-all excludes `api/`, `_vercel/`,
`assets/`, `new`, `exam`.

---

## 2. THE DESIGN SYSTEM

This is the section Robert cares about most. Internalize the workflow before writing any UI.

### The workflow — design on /design first, wire second

**Every piece of UI in the product is designed, demoed and reviewed on `/design`
(`src/ds/DesignPage.tsx`, ~2,080 lines) BEFORE it is wired into the app.** The page is the living
style guide AND the component workshop: every token, every component, every assembled screen is
rendered there in a `Spec` block (defined at `DesignPage.tsx:228` — a name, a note explaining the
design decision, and the live component with demo data).

The cycle:

1. Build the new component in `src/ds/` as a **token-only, props-driven** module — no fetching, no
   store, no auth. It renders entirely from props + `tokens.ts` values, so the style guide can demo
   the whole thing and the dark toggle flips it for free.
2. Add a `Spec` block (or a whole `SectionHeader` section) to `DesignPage.tsx` with realistic demo
   data (the page keeps fixtures like `DEMO_CANVASES`, `DEMO_STEPS`, `DEMO_VARIANTS`, `DEMO_COMMENTS`
   near the top).
3. Robert reviews it on `localhost:3000/design` — he judges by the rendered page, so screenshot it.
4. Only THEN wire it into `src/ds/app/` with real data hooks.

The `Spec` notes are not decoration — they record the reasoning ("media is bare — no card, no
header. the footage is the label") and are the design law for that component. Read them.

### Tokens — `src/ds/tokens.ts` + `src/ds/palette.css`

Every color in the system is a CSS variable defined in `palette.css`, exported as a TS constant in
`tokens.ts`. Light is `:root`, dark is `:root[data-theme='dark']`; `setTheme('dark'|'light')`
(`tokens.ts:135`) flips the attribute and every component follows with zero component changes.
Components must ONLY use these exports — a raw hex in a component is a bug.

- **Type: ONE family.** `FONT` = `'ppNeueMontreal'` (six weights loaded locally in `src/index.css`).
  `DISPLAY_FONT` and `MONO_FONT` are aliases of the same face — F37 Zagma is retired; the
  "mono" name survives only as the uppercase micro-label ROLE. Type scale objects (`display`,
  `displayLarge`, `eyebrow`, `cardTitle`, `body`, `microMono`) are measured off the published site.
- **Neutrals do all structural work:** `PAGE_BG` (#F6F7F8 cool studio ground), `CARD_BG` (pure
  white; **transparent in dark** — cards become pure outline, `CARD_BORDER` carries them),
  `MENU_BG` (opaque in BOTH themes — a portaled dropdown can never be see-through), `INSET_BG`,
  `INK`, `MUTED`, `HAIRLINE`/`HAIRLINE_SOFT`, `CHIP_BG`, `CARD_SHADOW`.
- **The strict six-role signal family** — named by MEANING, never by hue, all cool-shifted (no warm
  color exists in this system): `SIGNAL_BLUE` (visual peak / primary series), `SIGNAL_GREEN` (keep /
  buy moment / high), `SIGNAL_RED` (change / manipulation / low), `SIGNAL_VIOLET` (audio peak),
  `SIGNAL_STEEL` (moderate / caution — `SIGNAL_AMBER` is a deprecated alias of it), `SIGNAL_GREY`
  (scene change / neutral). A new case takes one of these roles or stays neutral.
- **Two stops per role:** the vivid `SIGNAL_*` stop is for GRAPHICS (diamonds, chart strokes,
  meters); the deepened `INK_*` stop (`INK_BLUE`, `INK_GREEN`, …) is for SMALL TEXT — the vivid blue
  is only ~3.4:1 on the ground, illegible at 13px. In dark, `INK_*` resolves to the vivid values
  (they already clear 7:1 on black).
- **`GLOW` — the electrode glow, the signature.** `#00C2FF` cyan (with `GLOW_CORE`, `GLOW_HALO`).
  The ONLY thing in the system allowed to look lit: live states, active nodes, the running
  playhead. **Never a data color, never set text in it.** `LIVE` aliases it.
- **`SIGNALS`** (`tokens.ts:93`) — the five markers the product tracks (buy_moment, visual_peak,
  audio_peak, scene_change, manipulation) with their color + ink, in one array so every legend on
  every surface labels and colors them identically. Legends read from here, never hand-roll one.
- **Radii:** `R_CARD = 20`, `R_INSET = 12`, `R_PILL = 999`. **Dot grid:** `DOT_GRID` + `DOT_SIZE`
  ('22px 22px') — the canvas ground.
- The `ds` object at the bottom bundles everything for `ds.CARD_BG`-style access.

`palette.css:116` — the document body follows the palette ONLY once `data-theme` is explicitly set
(DS pages set it on mount via `Page` in `components.tsx`); legacy dark surfaces (marketing homepage,
old MaryWorkspace) never set it and keep their own black.

### Icons — `src/ds/icons.tsx`

ONE library: lucide-react. Every glyph is imported at the top, pre-bound to the system defaults
(16px, stroke 1.5, `currentColor`) via the `bind()` wrapper (`icons.tsx:45`), and exported on the
`Icon` object (`Icon.play`, `Icon.brain`, `Icon.trash`, …).

**THE RULE (verbatim from the file): never an emoji, never a one-off inline SVG, never a second
icon pack.** Emoji render differently per platform and carry uncontrolled color, breaking palette
and dark mode at once. If a glyph is missing, import it from lucide and ADD it to this file so the
next person finds it — do not reach past this file. (The DS-internal transport icons in
`mediaAnalysis.tsx` predate the rule and are the one sanctioned exception — the DS core carries no
icon dependency.)

### The component files — what lives where

All in `src/ds/`, all token-only and props-driven:

| File | What it holds |
|---|---|
| `components.tsx` | The base grammar: `Card`, `Eyebrow`, `CardTitle`, `StatNumeral`, `InsetNote`, `SectionHeader`, `TopNav`, `Page` (sets the theme attribute on mount) |
| `controls.tsx` | `Button` (primary/secondary/subtle/danger, sm/md/lg, loading/disabled/icon), `IconButton`, `SegmentedControl`, `TextField` (56px target, inset fill, no border until focus/error), `Checkbox`, `Switch`, `Select`, `Slider`, `TextLink`, `Badge`, `Spinner`, `FieldLabel` |
| `overlays.tsx` | `Tooltip`, `Popover`, `DropdownMenu` (actions not values; arrow keys/Enter/Escape), `Modal`, `ConfirmDialog`, `ToastProvider` + `useToast()` → `push({tone, title, description})` — bottom-right, stacked, capped at three |
| `keyMetric.tsx` | `KeyMetricCard` — THE canonical metric format (fixed two-line LABEL slot → weight-200 numeral → tier word in its ink → sparkline; the fixed slot keeps a whole strip on one baseline). `TIERS` (strong/neutral/weak/none), `tierFromWord()`, `KeyMetricsGrid` (+ its new `bare`/`compact` props), `KeyMetricsCompareGrid`. Formats: percent · signed · score · raw; states: ready · scanning · loading · not measured |
| `mediaAnalysis.tsx` | The media-analysis instrument — three cards that are ONE object because they share a single playhead (the caller holds `sec`, nothing here owns time): **`VideoStageCard`** (draws frame/transport/scrub, media passed as children, `mediaAspect` letterboxes portrait vs landscape correctly, `muted`/`onMutedChange` controlled sound, `fill` drops the aspect lock, **`bare`** drops the card entirely — the picture sits on the page), **`LiveResponsePanel`** (traces drawn whole-series-faint + played-portion-strong — "the shape is the product"; `compare` ghosts a second scan resampled onto the same x-axis; also takes `bare`), **`StimulusTimelineCard`** (clickable diamonds seek; legend from `SIGNALS`), plus `LiveChip`, `usePlayhead` (rAF driven but committed at ~12fps — 60fps re-renders once caused a New Relic beacon storm, ~12k failed requests in 30s), `defaultTraces`, `markersFromRich` (derives markers from a real scan — manipulation from the real series, scene changes only from transcript segments, never invented), `fmtClock`, and the internal `Shell`/`BareFrame` pair that implements the `bare` pattern |
| `brain/TransparentBrain.tsx` | The WebGL cortex (Three.js via `brainEngine.ts`). One card: stage + activation scale + layer toggles + tuning knobs. Renderer is transparent and ADDITIVE — it needs a dark ground. `variant="full"` is the card; `variant="bare"` floats the cortex straight on the page (used by the ad-detail hero). Lazy-mounted; presets in `brainPresets.ts` are hand-authored demo values and say so on their face — real data enters at `handle.setSeeds()` |
| `adLibrary.tsx` | The public library wall: `NeuralAdCard`, `NeuralAdGrid`, `AdShelfHeader`, the `NeuralAd` type and `scoreTier()` |
| `adDetail.tsx` | **UNCOMMITTED (new)** — the neural-ad detail page (§8): `AdTitleBar`, `AdInfoPanel` (Information · "Scanned by" · Response mix · the scan-your-own funnel), `ScanModelChip` (portrait image with brain-glyph fallback, awaiting Robert's model-portrait library), `HowItWasMade` (miniature canvas snapshot with `SnapChip`s), `AdComments` + `AdCommentsTeaser` (the folded strip that unfolds in place), `AdRecommendedRail`, `ActionPill`, `DEMO_COMMENTS`, `DEMO_RESPONSE_MIX` |
| `canvasHome.tsx` | The canvas library wall: `CanvasCard` (poster thumb, score + tier, delta, running state, hover ⋯ menu with rename/duplicate/cover/delete), `CanvasGrid` (the + `NewCanvasTile` is FIRST, not last), `CanvasSummary` type |
| `canvas/` | The node language: `nodeShell.tsx` (`NodeShell` — status/icon/title/run/expand/⋯/ports/footer, `NODE_W`), `wire.tsx` (`Wire` — "THE differentiator: a measured score rides the edge", `GhostWire`), `canvas.tsx` (`Canvas`, `CanvasToolbar`, `WireLayer`, `useCanvasViewport`), `nodes.tsx` (`CanvasMedia` — bare, no shell, carries its own resize grip + new uploading/progress states; `CanvasMediaDrop` — the drop plate, now with the paste-a-link row; `AnalysisNode`, `ComparisonNode`, `PromptNode`) |
| `agent.tsx` | The agent surface: `AgentPanel`, `AgentTraceDock` (says what it is doing, not "processing"), `SegmentBand` (measured damage above, proposed fixes below, one axis), `VariantCard`, `AgentObjective`, `AgentNodeBody`, `VARIANT_KIND`, `GEN_MODELS`, `OBJECTIVES` |
| `expand.tsx` | `NodeExpand` (the full-screen overlay a node expands into), `ScanReadingLayout`, `ABSwitch`, `ABFrame` |
| `fingerprint.tsx` | `FingerprintStrip`, `FingerprintCard`, `BrainSummaryRow` |
| `scanRail.tsx` | `GradeScoreCard` (the grade is the hero), `FixListCard` (rows are controls — click seeks), `fixItemsFromMoments` |
| `account.tsx` | `Avatar` (initials fallback on a deterministic neutral), `UserMenu` (sign out separated and last), `AccountSummary` |
| `settings.tsx` | `SettingsLayout`, `SettingsNav`, `SettingsSection`, `FieldRow`, `SaveBar`, `PlanCard`, `PaymentMethodRow`, `ApiKeyRow` (masked at rest — copy never requires reveal), `DangerRow` (red is the border and the verb, never the fill), `UsageRow` |
| `apiPanels.tsx` / `apiDialogs.tsx` | `StatTile`, `UsageBars` (bars not a line — daily spend is discrete), `EndpointList`, `EnvBadge`, `ConnectAgentCard`, `DocSection`; `AddCreditsDialog` (a dialog that spends money says so), `NewKeyDialog`, `BalanceMeter` |
| `appShell.tsx` | `AppBar` (logo · tabs · user menu — never changes shape between surfaces), `AppShell`, `SapientMark`, `PageHeader`, `APP_BAR_H` |
| `auth.tsx` / `paywall.tsx` | `SignInForm`, `CreateAccountForm`; `PaywallScreen` |
| `chat.tsx`, `tabs.tsx`, `table.tsx`, `code.tsx`, `dropzone.tsx`, `feedback.tsx`, `runButton.tsx` | `ChatComposer`/`ChatMessage`; `Tabs` (pill/outline/underline); `Table` (uppercase headers, tabular figures, no zebra); `CodeBlock` (always dark); `FileDropzone`/`FileRow`; `Skeleton` (compositor-only shimmer)/`EmptyState` (an empty state without an action is a dead end); `RunButton`/`RunDot` |

### Recurring design laws (they come up in review)

- **The `bare` pattern:** a video (or a trace panel next to one) does not need a frame drawn around
  it — it already has hard edges. `bare` swaps the card `Shell` for `BareFrame` (a plain div,
  `mediaAnalysis.tsx:320`) so the media sits directly on the page. The library's own rule: nothing
  is ever overlaid on the artwork; the read lives beside it.
- **GLOW only for genuinely live states.** A `LiveChip` is a status, not decoration.
- **Honest progress:** when a transfer reports real progress, show a bar; when it does not (a link
  fetch), show an indeterminate shimmer — never a made-up percent. (See `nodes.tsx` `CanvasMedia`
  uploading state and §6.)
- **Legends and tier words come from `SIGNALS` / `TIERS`, never hand-rolled**, so no two surfaces
  can drift.
- **Cards clip themselves:** `minHeight: 0` + `overflow: hidden` so a card in an `fr` grid never
  pushes siblings off screen.

---

## 3. The canvas app (/app)

`src/ds/app/AppPrototype.tsx` — "the application, assembled from the design system and nothing
else." Despite the historical name "prototype", **this IS the shipped product**: it renders inside
a real `ClerkProvider` (main.tsx:262), enforces entitlement, and talks to the real API.

### Shell and surfaces

One `AppBar` over four surfaces (`AppPrototype.tsx:62`): tabs `home | api | docs`, plus `account`
as a surface reached from the user menu (account is a Surface, not a Tab — both are set together at
`:263` because the tab renders first). The bar never changes shape; an open canvas puts its name +
a back arrow in the bar's `left` slot with inline rename (`CanvasNameField`).

Gate order in `AppInner` (all before the product renders):
1. `!session.ready` → blank shell (never flash sign-in at someone already signed in).
2. Signed in + `access === 'checking'` → blank shell. **Fail closed** — the entitlement check used
   to be raceable and a brand-new account could use the app while the lookup ran
   (`useAccessGate`, `data/useAccessGate.ts`).
3. Signed in + `access === 'paywall'` → `PaywallScreen` with the real plans from
   `src/lib/plans.ts` (`sellablePlans`) and real Stripe checkout — the user is NOT bounced to the
   old `/paywall` route. Reads `sessionStorage['sapient:intendedTier']` to pre-select the plan
   clicked on the marketing pricing page ('agents' maps to internal tier key 'founders').
4. Signed out (or `/app/signin` / `/app/signup`) → `AuthGate` → `AuthScreen`, driven by the real
   Clerk email-code flow in `data/useAuthFlow.ts`.
5. Otherwise: Docs / API / Account / Canvas / Home.

Stripe returns land on `/app?upgraded=1&session_id=…&tier=…` (subscriptions) or `?topup=success`
(wallet). The handler at `AppPrototype.tsx:127` toasts, re-reads the wallet (the entitlement moves
when the WEBHOOK fires, not when the browser returns), fires the browser-side Meta Pixel `Purchase`
with the Stripe session id as `event_id` (dedupes against the server-side Conversions API event),
and strips the query so a refresh does not re-announce a purchase.

### Identity — `data/session.ts`

`useSession()` wraps Clerk. In production, identity is the Clerk session. On a dev build with
`VITE_CLERK_BYPASS=true`, `vite.config.ts` aliases `@clerk/react` → `src/lib/clerkShim.tsx` and a
local stand-in session carries the real Clerk user id (`DEV_USER_ID`, overridable via
`VITE_CLERK_BYPASS_USER_ID`) so every server call hits the same account the live app would. The
branch is gated on `import.meta.env.DEV` and compiled out of prod. `authHeaders(userId)` produces
either a real Clerk JWT `Bearer` or `Bearer dev-bypass-<userId>` (accepted server-side only when
`CLERK_BYPASS=true`, see §5). Session also carries `plan`, `isPro`, and `managed` — a
founder/comp grant that reads as pro with NO Stripe subscription behind it; billing hides Cancel
and routes plan changes to Checkout instead of the portal for these (the portal would 404).

### The home wall

`HomePage.tsx` renders `CanvasGrid` over a merged list (`AppPrototype.tsx:189–193`): saved cloud
canvases first (`useCanvases` → `canvasesToCards`), then any loose scan from `useScans` whose
`sessionId` is NOT already a canvas on the wall (a scan run FROM a canvas carries the canvas id as
its session — without the filter it listed twice). Loose scans have ids starting `qualia_run_` /
`mary_run_`; renaming one routes to `/api/qualia-run?action=rename` while a canvas renames through
`/api/mary-chats?action=rename` (`commitRename`, `:212`). Duplicating a loose scan is refused with
a toast — there is no board to copy. Card covers: one hidden file input for the whole wall; a
chosen cover uploads through `uploadCanvasFile` and is stored with the `cover:` prefix (below).

### The canvas — `CanvasPage.tsx` (~2,050 lines)

The node graph. Four rules, from the file header: media is BARE (no card — the footage is the
label); every runnable node owns its run (no global "run the graph"); THE WIRE CARRIES THE NUMBER
(`64 → 71 +7` rides the edge between two scans); the user builds the graph, nothing is
auto-arranged.

Types (`CanvasPage.tsx:45`):

```ts
type NodeKind = 'media' | 'analysis' | 'comparison' | 'prompt' | 'variant' | 'clip';
interface GraphNode {
  id, kind, x, y, label?,
  state?: RunState,                       // analysis/comparison/prompt
  runId?: string | null,                  // a FIELD, not the node id — re-keying would rewrite wires
  statusText?, startedAt?,                // elapsed clock derives from startedAt, not an interval
  score?, grade?, weakAt?, weakLabel?, progress?, elapsedSec?,
  text?, answer?,                         // prompt
  durationSec?, videoUrl?, thumbUrl?, aspect?,
  storagePath?, posterPath?,              // the DURABLE addresses; signed URLs die in 12h
  a?, b?,                                 // comparison sides {label, score, runId?, file?}
  objective?, steps?, weak?, variantIds?, // agent
  variant?,                               // one AgentVariant (variant/clip nodes)
  uploading?, error?,                     // media in flight / whatever went wrong, in the user's words
  width?, height?,
}
interface GraphWire { from, to, port?: 'a'|'b', label?: { text?, from?, to? } }
```

Key mechanics:

- **Uploads:** drop a file on `CanvasMediaDrop` → `uploadCanvasFile` (`src/agents/canvasUpload.ts`)
  → `/api/analyze?action=upload-url` mints a signed-upload token → **tus resumable upload**
  (`tus-js-client`, 6MB chunks, endpoint `${SUPABASE_URL}/storage/v1/upload/resumable`) straight
  into the **`kairo-uploads`** bucket → `?action=get-signed-url` returns a 12h READ url. The node
  stores BOTH `videoUrl` (ephemeral) and `storagePath` (permanent). Posters are captured in the
  browser (`data/poster.ts` — frame at ~0.5s, 1080px long edge JPEG, wholly best-effort: every
  failure resolves null, a poster must never block an upload) and uploaded through the same path
  as `posterPath`.
- **Pasted links:** `addMediaFromUrl` (`CanvasPage.tsx:1589`) — one pasted link becomes one media
  node. The node immediately shows `label: "Fetching from <host>…"`, `uploading: true`,
  `progress: 0` (deliberately: a link fetch has no real percent, so the plate shows the honest
  indeterminate shimmer). POST `/api/ig-download {url}` → on success the node gets
  videoUrl/storagePath/name — indistinguishable from a dropped file — and if the canvas is still
  "Untitled canvas" it takes the clip's name; a poster is captured from the remote URL
  (`capturePosterFromUrl`, needs CORS — Supabase signed URLs allow it).
- **Re-signing:** on hydrate and before any run, `resignPaths(paths)` (one request per board, via
  `/api/analyze?action=sign-paths`) re-mints URLs from storage paths. A null means "the file is
  gone", not an error — losing a video costs you the video, not the canvas. `freshSourceUrl`
  (`data/useRun.ts:66`) applies the same rule per node before a scan submits.
- **Running a scan:** `data/useRun.ts`. `submitScan` POSTs `/api/qualia-run?action=submit` with
  `{user_id, capability:'brain_map', modality:'video', analysis_mode:'full',
  input:{kind:'url', value}, session_id: <canvas id>, duration_sec}` (the duration hint is the
  cheap reject — refuse an over-cap clip before spending a GPU). Network throws are retried twice
  with backoff; 4xx/5xx are NOT retried (real answers — the run may already exist). The returned
  `run_id` is written onto the node BEFORE polling starts, so navigating away never orphans a
  running (billed) scan — any later mount re-attaches via `pollScan` (polling is a read).
  `pollScan` hits `?action=results` every 3s for up to 20min, mapping phases
  queued/extracting/predicting/rendering to progress + human copy ("Reading the frames"…). Failure
  copy never leaks plumbing; server messages written for a person (`video_too_long`,
  `insufficient_credits` → upgrade flag) are used verbatim. It deliberately does NOT reuse
  `src/agents/pipeline.ts` `runRealScan` (collapses every failure to null) or `useScanView` for
  polling (treats the 202-while-running as permanent error).
- **Agent / variants:** a prompt node with an analysis input becomes an agent
  (`proposeVariants` in `data/useAgent.ts`, generation via `data/useGenerate.ts` →
  `/api/generate-video` etc., models listed in `agent.tsx GEN_MODELS`). Until `/v1/agent/propose`
  is fully wired, `variantsFor` (`CanvasPage.tsx:197`) maps the scan's REAL weak windows to fitting
  changes — the objective FILTERS which seconds get worked on (an objective naming a signal pulls
  that signal's windows; "hold attention" and "drive sales" work on different seconds), and never
  invents a timestamp. Agent capability is gated by tier (`canUseAgent`,
  `src/lib/entitlements.ts`).
- **Reading views:** expanding an analysis node opens `ScanReadingView.tsx` (stage + traces +
  timeline + rail on one playhead) inside `NodeExpand`; a comparison opens `ScanCompareView.tsx` —
  "two variations of ONE view", using the `compare`/`compareMarkers` props of the same instruments.
  Scan artifacts become view models through `data/scanView.ts buildScanView`.

### Persistence — `data/useCanvases.ts` + `/api/mary-chats`

A canvas is a saved graph in the PROVEN chat-session storage (no new table):

- **Rows:** one `chat_sessions` row per canvas, namespaced by user id — `user_id =
  '<clerk_user_id>::canvas'` (the `kind` column has a CHECK constraint that rejects new values, so
  the namespace lives in user_id — `api/mary-chats.ts:29`). The graph itself is ONE
  `chat_messages` row whose content starts with `{"v":` (chat-log rows coexist and never match).
- **Actions** (`/api/mary-chats?action=…`): `list` (kind=canvas), `create`, `canvas-save` (upserts
  the graph row, 1MB cap, refreshes the session `preview`), `canvas-load`, `canvas-cover`,
  `canvas-duplicate` (server-side copy of session row + graph row, ownership-checked, rolls back
  the session row if the graph copy fails), `rename`, `delete`, plus the plain chat
  `messages`/`save`. Service-role Supabase client (bypasses RLS; the browser anon client only
  satisfies the Clerk-JWT RLS in prod — this is what makes canvases work on localhost too).
- **The preview column encodes the card art AND stats:** `poster:<storage path>#n=<nodes>&s=<best>`
  — a PATH, never a URL (signed URLs die in 12h; a wall of dead thumbnails is worse than none),
  with node count and best score riding after the `#` so the wall renders from the list query alone
  (no per-card graph fetch for two integers). A user-chosen cover uses the `cover:` prefix; the
  server keeps a held `cover:` picture through subsequent `poster:` autosaves while still updating
  the `#` stats (`api/mary-chats.ts:120–126`).
- **Autosave:** `useCanvasGraph` — load once, then debounced save 1.2s after the last change,
  fingerprinted so an unchanged board never writes. `scrub()` strips `blob:`/`data:` URLs and
  >64KB strings before persisting (a blob dies with the tab; a data-URL video would blow the row) —
  a node that loses its media keeps everything else and shows empty, which is honest.

Hooks live in `src/ds/app/data/`: `useCanvases`/`useCanvasGraph`, `useRun`, `useScans` (the
account's run history via `/api/qualia-run?action=runs`, UNIONed server-side with historical
`mary_runs`), `useScanView`, `useAgent`, `useGenerate`, `useBilling`, `useApiAccount` (keys +
wallet via `/api/sapient/*`), `useAvatar` (`/api/account/avatar`), `usePreferences`
(`/api/preferences`), `useAccessGate`, `useAuthFlow`, `useViewport`, `session.ts`, `poster.ts`,
`scanView.ts`.

---

## 4. The scan pipeline

The current pipeline is **Qualia**. Four generations coexist (see CODEBASE-MAP §3 for the other
three — legacy `api/analyze.ts`/`kairo_jobs`, `api/sapient-scan.ts`/`sapient_runs`,
`api/mary-run.ts`/`mary_runs`); new work goes through Qualia only.

**The full trace (video):**

1. Client (`useRun.submitScan` or `src/lib/maryWorkspaceClient.ts` from the old workspace) →
   `POST /api/qualia-run?action=submit`.
2. `api/qualia-run.ts` (1,084 lines): validates capability/tier (the "$1 unlock" entitlement gate
   mirrors `api/mary-run.ts`), cheap-rejects over-cap clips off the client `duration_sec` hint
   (`video_too_long`, no GPU, no credit), reuses an identical recent run when one exists
   (`findReusableRun`), INSERTs a `qualia_runs` row (id `qualia_run_<ts>_<rand>`, status queued,
   `session_id` = the canvas id), consumes one scan credit (refunded exactly and idempotently if
   the run terminally fails — `refundCreditIfTerminal` → `refundScanUnified`), resolves the input
   to a signed URL, and POSTs the Modal trigger with `{auth_token: PIPELINE_SECRET, run_id, …}`.
   `PIPELINE_SECRET = MARY_PIPELINE_SECRET || KAIRO_PIPELINE_SECRET` (`qualia-run.ts:75`).
3. **Modal.** The live serving workspace is **`thesapientcompany`** — app **`sapient-qualia-serve`**,
   function `submit` → `https://thesapientcompany--sapient-qualia-serve-submit.modal.run`.
   Resolution order (`api/_modelManifest.ts:186–189`): `QUALIA_SERVE_TRIGGER_URL` env override,
   else that hardcoded company-workspace URL. **The PR #132 story:** the fallback used to be
   templated off `MODAL_WORKSPACE_PRIMARY` (default the personal workspace `robert-16572`), which
   was EMPTIED on 2026-08-10 for cost cleanup — everything still pointing there 404'd with
   `modal_trigger_failed` (the 8/10 outage). PR #132 (merged 2026-08-21 as `6ff6efa`) cut the code
   fallback over to the company URL; a 422 on an empty POST is the "app is alive" probe. All
   serving functions are `min_containers=0, scaledown_window=300` — **scale-to-zero: the first scan
   after idle pays a ~1–2 min cold start; a normal video takes ~60–150s** end to end. Serving code
   lives outside this repo at `~/Desktop/Sapient-models/qualia/modal/serve.py`.
   **Law: never delete or rename Modal apps without `grep -rn "modal.run" api/` first** — prod
   points at workspaces by NAME and fails closed. Current referencing files: `api/analyze.ts`,
   `api/mary-run.ts`, `api/sapient-scan.ts`, `api/_modelManifest.ts`, `api/admin/pipeline.ts`,
   `api/sapient-intelligence/score-creatives.ts`.
4. The Modal worker writes status/progress/result straight into `qualia_runs` (service role); the
   client polls `GET ?action=results` (202-style status until terminal). Artifact →
   `buildScanView` (canvas app) or `interpretNeuro()` → `NeuroRead` (legacy result surfaces).
5. Other qualia-run actions: `runs` (history; UNIONs historical `mary_runs` rows with
   `model='qualia'`), `set_saved`, `rename` (`qualia-run.ts:1069–1075`).

**Link ingestion — `api/ig-download.ts`:** POST `{url}` → **yt-dlp** (gallery-dl fallback for
Instagram) downloads the clip — Instagram/TikTok get login cookies from the local Chrome profile
(`IG_COOKIES_FROM`, default `chrome:Profile 7`; retried once cookieless for public links) — then
uploads the mp4 to `kairo-uploads` at `agents-canvas/<sha1(url)[:16]>.mp4` and returns
`{videoUrl (2h signed), name, path}`. The sha1 path doubles as a cache: a re-pasted link returns a
fresh signed URL for the existing object without re-downloading (`cached: true`). Requires yt-dlp
installed on the host — on Vercel this returns 501 `no_downloader` unless the runtime has it; it is
primarily exercised through the dev server / environments where yt-dlp exists.

**Non-video (image/PDF/text):** routed to `/api/mary-run`, scored in-handler by
`api/_neuralVision.ts` (`estimateFingerprint` — Claude vision → the same artifact shape, plus
imageHotspots / heatmapPhrases / per-page PDF reads). Always `estimated: true`. See
`TURING-HANDOFF.md` §1 for the full write-up. Honesty rule: these are ESTIMATES and the UI says so.

---

## 5. The API layer

`api/` — every non-underscore file is a Vercel serverless function; `_*.ts` are shared helpers
(never routed). vercel.json rewrites `/v1/*` → `/api/v1/*`, `/mcp` → `api/mcp-endpoint.ts`, and the
`.well-known/oauth-*` pair → the oauth metadata handlers. Crons: `refresh-cache` (daily),
`raise/snapshot` (daily), `observability-push` (*/10), `run-watchdog` (*/15). maxDurations:
analyze 60s, mary-run 60s (+ bundled pdfjs worker), observability-push 60s, run-watchdog 30s.

**Auth:** Clerk, verified server-side by `requireClerkUser` in `api/_sapient_helpers.ts:359`. When
`CLERK_BYPASS=true` (local env, NEVER prod) it accepts `Bearer dev-bypass-<userId>` without a
Clerk round-trip — the counterpart of the frontend `VITE_CLERK_BYPASS` clerkShim alias.
`SAPIENT_DEV_TIER` (same file, `:142`) can force an entitlement tier locally. Admin routes use
`api/admin/_guard.ts` (verified `__session` cookie). Public `/v1/*` uses bearer `sk_live_…` API
keys resolved to an org via `memberships` (prefix + bcrypt `secret_hash` in `sapient_api_keys`).
**There is no Clerk webhook handler.**

The endpoints, grouped (what the canvas app actually calls is marked ★):

- **Scans:** ★`qualia-run.ts` (submit/results/runs/set_saved/rename — §4), `mary-run.ts` (vision
  estimates + legacy), `analyze.ts` (3.1k-line legacy v1; ★ still owns the upload plumbing:
  `upload-url`, `get-signed-url`, `sign-paths`), `sapient-scan.ts` + `sapient-scan/[action].ts`
  (v2, legacy), ★`ig-download.ts`, `batch-link-job.ts`/`batch-results.ts`, `free-scan-status.ts`,
  `run-watchdog.ts` (cron: times out stuck runs).
- **Canvas/chat:** ★`mary-chats.ts` (§3 persistence), `mary-chat.ts`, `mary-workspace-chat.ts`,
  `mary-content-analysis.ts`, `chat.ts` (Anthropic + Supermemory), `chat-feedback.ts`,
  `mary-ripple.ts`.
- **Generation/agent:** `agent-plan.ts`, `generate-video.ts`, `generate-image.ts`,
  `generate-shot.ts`, `enhance-video.ts`, `assemble.ts`, `extract-frame.ts`, `filmstrip.ts`,
  `thumbnail.ts`, `transcribe.ts`, `tts.ts` (helpers: `_fal.ts`, `_higgsfield.ts`, `_ffmpeg.ts`,
  `_keyframes.ts`).
- **Billing (Stripe, LIVE):** `checkout.ts`, `billing-portal.ts` (dual account — primary +
  `STRIPE_SECRET_KEY_LEGACY`), `check-subscription.ts`, `credits.ts`,
  `sapient/stripe-webhook.ts` — **the ONLY webhook verifier** (idempotency via
  `sapient_processed_events`). Two money systems, never mixed: scan credits
  (`credit_overrides` + `analysis_credits`) vs the API dollar wallet (`sapient_deposits`) — see
  `src/lib/scanPacks.ts` header. **Two-tier pricing state:** `src/lib/plans.ts` defines
  `model` = "Sapient Model" $19.99 and `founders` = "Sapient Agents" $49.99 (the `founders` key is
  the historical internal tier name; the `agent` capability hangs off it). Prices resolve from env
  (`priceEnvVar`: `STRIPE_MODEL_PRICE_ID`, `STRIPE_FOUNDERS_PRICE_ID`); LIVE prices exist
  (`price_1U6v8r2SGpkiMY8NBKqASIAK` / `price_1U6v8s2SGpkiMY8NdRoE0kby`, creation script
  `scripts/create-tier-prices.ts`) but as of the last handoff **Robert still had to set/repoint
  those two vars in Vercel prod** — until then Model checkout errors cleanly and Agents charges the
  old price. Legacy $29/$49 subs are grandfathered via `LEGACY_FOUNDERS_PRICE_IDS`. Older
  Pro $149 / Studio $299 / Scale $2500 tiers remain defined for existing customers.
- **API wallet + keys:** `sapient/usage.ts`, `sapient/keys.ts` (+`keys/`),
  `sapient/deposit-checkout.ts`, `sapient/deposits.ts`, `sapient/auto-refill.ts` — what
  `useApiAccount` consumes.
- **Account:** ★`account/avatar.ts`, ★`account/invoices.ts`, ★`account/delete.ts`,
  ★`preferences.ts`, `profile.ts`, `users/`.
- **Email:** Resend lane in `api/email/` — `welcome.ts`, `scan-complete.ts`, `resend-webhook.ts`
  (Svix-verified), `_templates.ts`. (AgentMail/outbound-GTM lanes live outside this repo.)
- **Public API `/v1`:** `v1/analyze.ts` (+`analyze/`), `v1/scans.ts` (+`scans/`),
  `v1/intelligence.ts`, `v1/wallet.ts`, `v1/demo.ts` (public), `v1/case-study.ts`,
  `v1/parcellation.ts`, `v1/agent/`. Shapes in `_apiResult.ts`/`_scanShape.ts`; org memory in
  `_memory.ts` (Supabase `sapient_user_scans` + Pinecone). MCP: `mcp-endpoint.ts` + `api/mcp/` +
  the published `packages/sapient-mcp` (the `mcp__sapient__*` tools), OAuth metadata endpoints
  beside it.
- **Ops/admin:** `admin/*` (users/pipeline/funnel/waitlist/keys/settings/api-monitoring),
  `ops/`, `observability-push.ts` (New Relic cron), `events.ts`, `health.ts`, `refresh-cache.ts`.
- **Investor raise:** `api/raise/*` (tokenized `ivt_` dashboards, dataroom, deck, snapshot cron) —
  paired with `src/raise/*`.
- **Misc:** `library.ts` (the public ad library), `brain-exam.ts`, `brain-preorder.ts`,
  `business-access.ts`, `careers/`, `share/`, `sapienteval/`, `labs/`, `insights.ts`, `score.ts`,
  `signal.ts`, `webhooks/`, `demo-login.ts`, `validate-bonus-code.ts`, `component-notes.ts`.
  Legacy ad-intelligence under `sapient-intelligence/`.

**Database:** Supabase project `tndmiyfklhhpmcfxlfxl`. The live-table inventory, the
stale-table graveyard, and the storage buckets are in CODEBASE-MAP §4 — the two-sentence version:
`qualia_runs` + `mary_runs` are the run rows, `chat_sessions`/`chat_messages` hold canvases,
`kairo-uploads` (79 GB, legacy name) is the LIVE primary bucket, and row-count estimates from
list-tables LIE — always `count(*)`.

---

## 6. Operational rules

The ones that bite. Non-negotiable.

1. **Pushing to `main` deploys production.** Never push without Robert's explicit word. Work on a
   branch; gate every change on `npx tsc --noEmit` AND `npm run build` clean; after a sanctioned
   push, watch `npx vercel ls` until `● Ready` and curl the live site. Verify deploys by grepping
   served CONTENT — the SPA catch-all 200s any path, so status codes prove nothing.
2. **Never commit secrets.** `.env.local` is gitignored (so are `file.env` and `.env*`) — keep it
   that way. The `ANTHROPIC_API_KEY` in local `.env.local` is a DEAD zero-spend key (401s by
   design). To test vision locally, pass a working key in the shell env at server start — never
   write it to any file. Same for Stripe keys (they are LIVE).
3. **Kairo is legacy NAMING on LIVE infra — never rename casually.** The `kairo-uploads` bucket
   (79 GB, the primary upload store the tus client, ig-download and Modal all write to), the
   `kairo_jobs`/`kairo_results` views (88 query sites), `KAIRO_PIPELINE_SECRET` (read with no
   fallback in two places), nine `kairo_*` localStorage keys, askkairo.com redirects. Each has a
   documented migration recipe in CODEBASE-MAP §8 and `docs/KAIRO-RENAME-AUDIT.md`. A find/replace
   here is an outage.
4. **Two Vercel accounts exist:** the Sapient team (`robert@thesapientcompany.com`, team
   `the-sapient-company` — this repo) and Robert's personal RRG account. Check which one you are
   linked to before any `vercel` CLI action. Also: the CLI account cannot write prod env vars or
   deploy the Modal `qualia-serve` app — those are Robert's; hand them to him, never fake them.
5. **Verify UI on localhost with real screenshots.** Robert judges by the rendered page, not the
   diff. For design work that means `/design` first (§2). A stale `node_modules/.vite` cache has
   repeatedly caused "my edit didn't take effect" — clear it when starting the server (§7).
6. **The honesty rule.** Report real errors with the real cause; never fake a number in a UI
   state. Concrete precedents baked into the code: the canvas Run button was once a six-second
   timer inventing a score (`useRun.ts` header — the thing this codebase is built to never do
   again); a link fetch shows an indeterminate shimmer, never a made-up percent (`nodes.tsx`);
   non-video scans are labeled ESTIMATES; failed scans say "Nothing was charged" only because the
   refund is real (`refundCreditIfTerminal`). Empty states are verdicts with actions, not blanks.
7. **Modal law:** never delete/rename Modal apps without grepping `api/` for `modal.run` first
   (§4). Scale-to-zero means a cold first scan is NORMAL — do not diagnose a 90-second wait as an
   outage.
8. **Money:** scan credits and the API dollar wallet are separate systems — never cross them.
9. **No new frameworks by reflex:** no React Router (routing is deliberate pathname gates + state),
   no Redux/Zustand, no second icon pack, no CSS files beyond the DS's `palette.css` and existing
   entries — inline styles from tokens are the DS's idiom.

---

## 7. How to run it

```bash
cd ~/Desktop/Sapient
npm install                       # once

# Kill the port + clear Vite's transform cache (stale cache = invisible edits):
lsof -ti:3000 | xargs kill -9 2>/dev/null; rm -rf node_modules/.vite

# Dev server — Express (server/index.ts) with Vite as middleware; everything on :3000
VITE_CLERK_BYPASS=true CLERK_BYPASS=true npm run dev
# npm run dev:watch  → auto-restart on server/index.ts changes
```

Env: `.env.local` (gitignored) holds Supabase/Stripe/etc. for local work; `.env.example` is
STALE — ~60 var names exist only in code (CODEBASE-MAP §5). The Clerk secret lives ONLY in Vercel
as a sensitive var and cannot be pulled. Extras you may need at start time, passed in the shell:
`ANTHROPIC_API_KEY` (a working one, from Robert, for vision tests), `STRIPE_MODEL_PRICE_ID` /
`STRIPE_FOUNDERS_PRICE_ID` (two-tier checkout), `SAPIENT_DEV_TIER` (force an entitlement),
`VITE_CLERK_BYPASS_USER_ID` (act as a specific user).

Where to look once it is up:

- `http://localhost:3000/design` — the style guide. Theme toggle top-right; every component demo.
- `http://localhost:3000/app` — the product. With the bypass flags you are signed in as Robert's
  account (real data, real scans — a Run spends a real GPU + credit; prefer short clips).
- `http://localhost:3000/home/index.html` — the static marketing homepage (dev parity route).
- `http://localhost:3000/library` — the public ad library.

Checks before any handoff/deploy: `npm run lint` (= `tsc --noEmit`) and `npm run build` — both
clean. There is no test suite wired into CI; the type-check and the build are the gate.

---

## 8. Current in-flight work (Aug 2026)

The working tree carries one uncommitted design round (16 modified files + the new
`src/ds/adDetail.tsx`, ~690 insertions). Two threads:

**A. The neural-ad detail page design pass** (`src/ds/adDetail.tsx` + the "Neural ad detail"
section of `DesignPage.tsx:1785–1910`). What opens when someone clicks a card on the library wall —
the Higgsfield-style video-platform detail page translated into this system, per Robert's sketch:

- **Video-first bare hero:** the ad is the first thing on the page and wears NO card —
  `VideoStageCard fill dense bare` with the media seamless against the background; the live cortex
  (`TransparentBrain variant="bare"`) and a `bare` `LiveResponsePanel` sit BESIDE it on one shared
  playhead. Nothing is ever overlaid on the artwork. The new `bare` props on
  `VideoStageCard`/`LiveResponsePanel`/`KeyMetricsGrid` (in `mediaAnalysis.tsx`/`keyMetric.tsx`
  diffs) exist for exactly this.
- Under the title bar: the six key-metric squares (`KeyMetricsGrid bare compact columns={6}`), then
  **`AdCommentsTeaser`** — the folded one-line comment strip that unfolds into the full
  `AdComments` thread in place — then `HowItWasMade` (the canvas snapshot). Right rail:
  `AdInfoPanel` (Information / "Scanned by" / Response mix / the scan-your-own funnel) +
  `AdRecommendedRail`.
- **`ScanModelChip`** credits the scanning model ("Qualia · 7 networks") with an image slot —
  currently falling back to the brain glyph, awaiting Robert's model-portrait library.
- Status: design pass only — everything renders from props + demo data on `/design` (the assembled
  page opens in a `NodeExpand` from any library card there). Wiring into `/library/<id>` comes
  after Robert signs off the design.

**B. The canvas paste-link feedback fix** (`CanvasPage.tsx` + `canvas/nodes.tsx` diffs). A pasted
link used to look like a no-op while the server fetched it. Now: `CanvasMediaDrop` grew an
`onLink` paste/Enter input row on the drop plate; `CanvasMedia` grew `uploading`/`progress` and
renders a live working state — dimmed glyph, a real progress bar when the transfer reports true
progress (file uploads do), an honest indeterminate shimmer when it does not (link fetches), with
the name row saying WHAT ("Fetching from instagram.com…"). `addMediaFromUrl` threads the states
and error copy through. Related diff ripples: `useRun.ts` (submit retry-on-network-throw),
`useCanvases.ts`/`mary-chats.ts` (`canvas-duplicate`), `poster.ts` (`capturePosterFromUrl`),
`canvasHome.tsx` (inline rename / cover / duplicate menu), `AppPrototype.tsx` (wall rename
routing), `account/invoices.ts`, `icons.tsx` (new glyphs).

Also open (needs Robert, not code): the two Stripe price-id env vars in Vercel prod (§5), and one
live end-to-end video-scan verification after the PR #132 Modal fallback cutover (§4).

---

## 9. Reading list

In order, after this file:

1. `docs/CODEBASE-MAP.md` — the maintained repo map: full route tables, module map, the four
   pipelines, DB inventory, external-service locations, deploy details, conventions, the Kairo
   rename backlog, Modal topology.
2. `TURING-HANDOFF.md` (gitignored, repo root) — the engineering seat's live operating context:
   the multi-modality vision system, the two-tier pricing rollout, current blockers.
3. `CLAUDE.md` — the condensed intro Claude Code sessions boot on (note: its "Project Structure"
   predates `src/ds/`; trust this file and the map for layout).
4. `docs/ARCHITECTURE.md` (NeuroRead data spine) · `src/verdict-v3/ARCHITECTURE.md` (legacy result
   components) · `docs/USAGE-BILLING-SPEC.md` · `docs/KAIRO-RENAME-AUDIT.md` ·
   `docs/AGENTS-CANVAS-SPEC.md` · `docs/SUPABASE-CLEANUP-PLAN.md`.
5. The `Spec` notes on `/design` itself — the design decisions are documented where they render.

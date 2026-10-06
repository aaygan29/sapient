# Sapient Architecture — NeuroRead is the Single Source of Truth

> **Canonical reference for how a model output becomes something a user sees.** For everyone — Robert, Aayush, and any Claude agent. Created 2026-06-12. Pairs with `sapient-models/WORKFLOW.md` (the cross-repo shipping process) and `sapient-models/handoff/capabilities.json` (the data contract).

---

# PART A — The simplified, high-level view (the whole thing in one breath)

**One rule:** *Every model output flows through `neurosignal` into one object — `NeuroRead` — and every component on the frontend reads from `NeuroRead`. Nothing reads the raw model output directly.*

```
   A VIDEO/IMAGE/TEXT
        │
        ▼
   ┌──────────┐      raw brain numbers     ┌─────────────────┐     ONE object      ┌──────────────┐
   │  MODEL   │  ───  (Yeo-7 networks  ───► │   neurosignal    │ ───  NeuroRead  ──► │  EVERYTHING  │
   │ Mary or  │        + optional           │  (the universal  │    (metrics +       │  the user    │
   │ Qualia   │         neuro block)        │   translator)    │     constructs +    │  sees        │
   └──────────┘                             └─────────────────┘     confidence)      └──────────────┘
   swappable                                 backend = Aayush       THE CONTRACT       components (you)
```

Three things can change. Only one ever costs you a new component:

| What changes | Does it just update on NeuroRead? | What you do |
|---|---|---|
| **New data** in the neuro block (e.g. a "Trust" metric) | ✅ Yes — appears in `NeuroRead.metrics`/`.constructs` | Nothing, *if* it reuses a display type. New SHAPE → build 1 component. |
| **New lens** (a new way to group/view) | ✅ Same NeuroRead, just a new grouping | Tag which components belong to the lens (config only). |
| **Model upgrade** (Mary retrained / new model) | ✅ Identical NeuroRead, better numbers | Nothing downstream. The model is swappable. |

**That's the entire design.** New *data* is free. New *display shapes* cost exactly one reusable component. The model behind the curtain can be replaced anytime.

---

# PART B — The detailed (complex) view

## B1. The data contract: what `NeuroRead` actually is

`NeuroRead` is produced by **one function** — `interpretNeuro(networks, serverNeuro, dynamics)` in `src/lib/neuroInterpret.ts`. It is the only translation entry point. Shape:

```
NeuroRead {
  metrics:    NeuroMetric[]     // Positivity, Intensity, Engagement, Buy Signal, Manipulation
  constructs: NeuroConstruct[]  // reward_value, emotion, attention, visual_sensory,
                                //   memory_encoding, conflict_risk, cognitive_load
  buySell:    number (0–100)
  recommendation: 'Buy' | 'Hold' | 'Sell'
  source:     'server' | 'client-mirror'   // 'server' = the real neurosignal block was used
  confidence: number (0–1)
  lowSignal:  boolean           // honesty flag — true when the model output is collapsed/flat
}
```

## B2. The flow, end to end (audited 2026-06-12)

```
┌─ 1. MODEL (sapient-models, Modal: qualia-serve / mary-serve) ──────────────────────┐
│  Emits an artifact (stored in Supabase mary_runs.result):                          │
│    networks            Yeo-7 means         (always)                                │
│    networkTimeSeries   per-second Yeo-7    (video/audio)                            │
│    kpiTimeSeries       10 per-second KPIs                                           │
│    neuro               ◄── the neurosignal block. OPTIONAL TODAY (not deployed yet) │
└────────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─ 2. THE ONE DOOR — interpretNeuro(networks, serverNeuro) ─ src/lib/neuroInterpret.ts ┐
│    IF artifact has a `neuro` block (serverNeuro.metrics present):                    │
│        → use it directly        source = 'server'    ◄── the goal state             │
│    ELSE:                                                                             │
│        → compute a client-side mirror from `networks`   source = 'client-mirror'     │
│    overallPull(networks) → the single headline 0–100 score (contrast-based)          │
│                                                                                       │
│    ───────────────────────►  returns NeuroRead  ───────────────────────────────────┤
└───────────────────────────────────────────────────────────────────────────────────┘
              │                          │                              │
              ▼                          ▼                              ▼
   ┌─────────────────┐        ┌────────────────────┐         ┌──────────────────────┐
   │ DASHBOARD        │        │ PUBLIC API          │         │ FRONTEND COMPONENTS   │
   │ (Results tab)    │        │ api/_apiResult.ts   │         │ src/verdict-v3/*      │
   │ AnalysisV2Preview│        │   → /v1/analyze     │         │ NeuroInsights,        │
   │                  │        │ api/_scanShape.ts   │         │ hero + concept cards, │
   │                  │        │   → /v1/scans       │         │ BrainPanel, curves    │
   └─────────────────┘        └────────────────────┘         └──────────────────────┘
   ALL THREE read NeuroRead. The API and the UI can never disagree — same source.
```

## B3. The display types (how NeuroRead becomes pixels)

A capability in `handoff/capabilities.json` declares a `display` type. Each type maps to a built component:

| `display` | Component(s) | Reads from NeuroRead | Modality |
|---|---|---|---|
| `dial` | NeuroInsights (headline reads) | `metrics[]` (score 0–100) | universal |
| `bars` | KpiCompareBars, pillars | `constructs[]` + kpiSummary | universal |
| `curve` | AttentionCurve | kpiTimeSeries (per-second) | video/audio |
| `brain_map` | BrainPanel | networks → Glasser regions | universal |
| `card` | ConversionRiskFlag, PurchaseIntentMeter, PersuasionEthicsFlag, lens heroes | `constructs[]` + `buySell` | universal/lens |
| `image` | VerdictMediaFrame | source_url (poster) | video/image |

## B4. Two gates decide if a component shows

A component renders only when **both** are true:
1. **Feature flag** — `GET /api/ops/component-config` (per model + lens; default = enabled).
2. **BlockVisibility** — modality gate (e.g. a `curve` needs a timeline → hidden on text/image).

The component registry — `src/verdict-v3/resultComponents.ts` — is the catalog: each entry has a `key`, `slot` (a–e), `visKey`, `scope`, `lenses`, `status`. The live playground to see/toggle them is **`/ops/components`**.

---

# PART C — The Component Generation Playbook (when, how, from what)

**This section answers: when do we build a component, how, and which NeuroRead data do we pull from?**

### WHEN to generate a component (the decision)

```
A new capability arrives in handoff/capabilities.json
        │
        ▼
Does an existing `display` type (dial/bars/curve/brain_map/card/image) already show it?
        │
   ┌────┴─────┐
   │ YES      │ → DO NOTHING. It auto-renders. (This is ~90% of cases.)
   │ NO (new  │ → BUILD ONE component (a new display shape), then it's reusable forever.
   │   shape) │
   └──────────┘
```

You only build when the data needs a *shape* none of the six display types can draw (e.g. a per-person brain switcher, a confidence/abstention band, a two-subject tug-of-war).

### HOW to generate a component (the steps)

1. **Pull the real data** to design against: the capability's sample `neuro` block in `sapient-models/handoff/samples/<model>/`. *(Today this folder is empty — a sample MUST exist before building; see the upgraded workflow rule below.)*
2. **Build the component** in `src/verdict-v3/` (concept card, hero, or block). It must take its input from a `NeuroRead` field — never from raw model output.
3. **Register it** in `src/verdict-v3/resultComponents.ts` (`key`, `slot`, `visKey`, `scope`, `lenses`, `status: 'experimental'`).
4. **Add it to the library** at `/ops/components` with the sample JSON so the whole team can see the render + the data that drives it.
5. **Promote** `status` to `stable` once verified in the running app against a real sample.

### WHICH NeuroRead field to pull from (the map)

| If the new capability is about… | Pull from NeuroRead field |
|---|---|
| A headline score (0–100) | `metrics[]` (match by key: positivity/intensity/engagement/buy/manipulation) |
| A brain construct | `constructs[]` (reward_value, emotion, attention, visual_sensory, memory_encoding, conflict_risk, cognitive_load) |
| Buy/Sell verdict | `buySell` + `recommendation` |
| Confidence / "we're not sure" | `confidence` + `lowSignal` |
| Per-second timeline | `kpiTimeSeries` (from the artifact, alongside NeuroRead) |
| Brain map | `networks` → region mapping (alongside NeuroRead) |

**Rule of thumb:** if you're writing code that reads the raw model artifact directly inside a component, stop — route it through `interpretNeuro` so it lands on `NeuroRead` first. That keeps the codebase clean: one translator, one vocabulary, one place to change.

---

# PART D — The upgraded workflow (what's new vs `WORKFLOW.md`)

The existing `WORKFLOW.md` process is correct but trust-based. These four rules make the NeuroRead contract self-enforcing so bad/undisplayable data can't move down the pipe silently:

1. **A real sample is mandatory.** Every capability in `handoff/capabilities.json` ships with a real `neuro`-block sample in `handoff/samples/<model>/`, or the PR is red. (Fixes today's empty-samples gap — you can't design against data that doesn't exist.)
2. **The model-handoff loop auto-audits** (see `OS/loops/model-handoff/`): on every push it checks sample-matches-schema, every-capability-has-a-component, every-component-has-a-capability, and client-mirror-agrees-with-server.
3. **neurosignal (backend) is canonical.** The client-side mirror inside `neuroInterpret.ts` is fallback-only and gets pruned once the real `neuro` block ships; a test asserts the two agree on the sample so they can't drift.
4. **`/ops/components` renders real samples,** not fake data — so the team always designs against the truth.

---

# PART E — Naming (settled 2026-06-12)

**The translation layer is called `neurosignal` everywhere.** It is the universal translator (Aayush's backend package, lives in `sapient-models`) that produces the `neuro` block. The app-side consumer is `src/lib/neuroInterpret.ts` (`interpretNeuro` → `NeuroRead`). The retired band-aids (`intelligenceTranslation.ts`, `TEMPINTELLIGENCETRANSLATION.py`) are being removed in favor of consuming the real neurosignal output.

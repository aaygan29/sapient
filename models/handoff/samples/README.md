# samples/ — real model output the frontend renders against

Drop the actual JSON a model produces here, one folder per model:
```
samples/
  mary/      # Mary's artifact JSON (Yeo-7 networks, per-second kpiTimeSeries, composites, verdict summary)
  qualia/    # Qualia (kairo-serve) output — already mapped into Mary's artifact shape by api/_qualiaRun.ts
```
Rules:
- **Real, validated output only — never mocked.** This is what the frontend uses to build + verify components, and what
  the `/ops/components` API-response library shows the team.
- Regenerate these whenever the model output changes (a new capability, better numbers) so the UI and the JSON stay in
  sync with what production actually returns.
- The matching contract (what each field means + how it should be displayed) lives in `../capabilities.json`.

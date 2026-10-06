# Sapient Intelligence API

Predict how real human brains respond to your content. Sapient's model is
trained on real human fMRI brain responses — you send a piece of content, you
get back a brain-grounded engagement read across five lenses.

There is **one endpoint**: `POST /v1/analyze`. You submit content, get a scan
object back, then poll it (or receive the result on a webhook).

Base URL: `https://www.thesapientcompany.com`

> Status: Beta. Fields may still change.

---

## 1. Authenticate

1. Sign in at [thesapientcompany.com](https://www.thesapientcompany.com) → **Settings → API**.
   You need a **paid plan ($19.99/mo, X1 or higher)** to create a key.
2. Click **Create key**. The secret (`sk_live_…`) is shown **once** — copy it now.
3. Send it as a Bearer token on every request:

```
Authorization: Bearer <YOUR_SAPIENT_API_KEY>
```

Test keys start with `sk_test_`. Lost a key? Revoke it in the portal and mint a
new one — secrets are never recoverable.

---

## 2. Submit a scan — `POST /v1/analyze`

**Request**

```json
{
  "input": { "type": "url", "url": "https://yourcdn.com/ad.mp4" },
  "webhook_url": "https://yourapp.com/hooks/sapient",
  "lens": "attention"
}
```

| Field         | Type   | Required | Notes                                                                 |
|---------------|--------|----------|-----------------------------------------------------------------------|
| `input`       | object | yes      | `{ "type": "url", "url": "<public asset url>" }` (video/audio/image) or `{ "type": "text", "text": "<copy>" }`. |
| `webhook_url` | string | no       | If set, we `POST { id, status, result }` here when the scan completes. |
| `lens`        | string | no       | Frame the read around one lens: `attention`, `emotion`, `memory`, `buy_sell`, `manipulation`. Display only — does not change the numbers. |

**Response (202 — async)**

```json
{ "id": "mary_run_abc123", "status": "queued" }
```

Some inputs (e.g. text or a still image) complete immediately and return
`200` with the result inline:

```json
{ "id": "mary_run_abc123", "status": "complete", "result": { … } }
```

---

## 3. Get the result — poll or webhook

### Poll: `GET /v1/analyze/{id}`

```json
{ "id": "mary_run_abc123", "status": "complete", "result": { … } }
```

`status` is one of `queued`, `processing`, `complete`, `error`. The `result`
object is present only when `status` is `complete`. A scan that isn't yours (or
doesn't exist) returns `404`.

### Webhook

If you passed `webhook_url`, we `POST` the same payload to it once on completion:

```json
{ "id": "mary_run_abc123", "status": "complete", "result": { … } }
```

The webhook is best-effort (8s timeout, fired once). Polling is always available
as a fallback, so charging never depends on the webhook reaching you.

---

## 4. The result shape

One shape, always:

```json
{
  "score": 78,
  "grade": "B",
  "lenses": {
    "attention": 72,
    "emotion": 64,
    "memory": 58,
    "buy_sell": 61,
    "manipulation": 47
  },
  "manipulation": {
    "conversion_risk": 82,
    "risk_band": "high",
    "cognitive_sovereignty_index": -1.88,
    "sovereignty": 18,
    "quadrant": { "key": "total", "label": "Total Capture", "lowers_guard": 67, "wins_over": 69, "how": "…" },
    "circuits": [ { "key": "evaluation", "label": "Critical thinking", "reading": "below normal", "value": -0.37, "meaning": "…" } ],
    "mechanisms": [ { "key": "SR", "label": "Social Reward", "share": 0.6 } ],
    "peak_moment": { "peak_sec": 15, "start_sec": 14, "end_sec": 16, "quote": "…", "why": "…" },
    "tactics": [ { "tactic": "Social proof", "kind": "everyone's doing it", "at_sec": 0, "quote": "…" } ]
  },
  "raw": {
    "networks": { "Visual": 0.13, "Limbic": 0.14, "…": 0.0 },
    "kpis": { "summary": { "…": {} }, "peaks": null },
    "composites": [ { "name": "Visual Pull", "score": 84, "tier": "strength" } ],
    "neuro": null,
    "score": 78,
    "grade": "B"
  }
}
```

- `score` — the headline 0–100 brain-engagement prediction.
- `grade` — letter grade for that score.
- `lenses` — five 0–100 reads (50 = neutral) derived from the brain networks:
  - `attention` — how much it holds focus.
  - `emotion` — emotional valence, re-based to 0–100.
  - `memory` — how likely it is to stick.
  - `buy_sell` — approach-vs-avoid buy signal.
  - `manipulation` — emotional pull vs reasoning.
- `manipulation` — **present only when you pass `lens: "manipulation"`.** The full
  manipulation read: `conversion_risk` (0–100) + `cognitive_sovereignty_index`,
  the two-axis `quadrant` (lowers-your-guard × wins-you-over), the four brain
  `circuits` (critical thinking / reward & approval / gut-instinct alarm /
  personal relevance, each with an above-or-below-normal reading), the
  persuasion `mechanisms` split, the most-persuasive `peak_moment` (with the
  transcript line + why), and the detected `tactics` (timestamped). Directional,
  network-proxied. The same block also rides on `GET /v1/scans/{id}`.
- `raw` — the underlying numbers: brain `networks`, trimmed `kpis`
  (summary + peaks, never the full per-second timeline), `composites`, the
  server `neuro` block (when present), and the `score`/`grade`.

---

## 5. Pricing

- **$1.00 per scan**, charged once when the scan completes. Failed scans are
  never charged.
- The **first $5.00 of usage each month is free**; after that, scans debit your
  prepaid deposit balance.
- Credits never expire. Add credits in **Settings → API → Add API credits**.
- A paid plan ($19.99/mo, X1 or higher) is required to create a key.

Current usage and balance are always visible in the portal (Settings → API).

---

## 6. Quick start (curl)

```bash
# Submit
curl https://www.thesapientcompany.com/v1/analyze \
  -H "Authorization: Bearer <YOUR_SAPIENT_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{ "input": { "type": "url", "url": "https://yourcdn.com/ad.mp4" } }'
# → { "id": "mary_run_abc123", "status": "queued" }

# Poll
curl https://www.thesapientcompany.com/v1/analyze/mary_run_abc123 \
  -H "Authorization: Bearer <YOUR_SAPIENT_API_KEY>"
# → { "id": "mary_run_abc123", "status": "complete", "result": { … } }
```

---

## Errors

Standard HTTP status codes. Error bodies look like:

```json
{ "error": "unauthorized", "message": "Missing, invalid, or revoked API key." }
```

| Status | Meaning                                                       |
|--------|--------------------------------------------------------------|
| 401    | Missing / invalid / revoked key.                             |
| 402    | Out of credits — add a deposit in the portal.               |
| 403    | Your plan can't use the API — upgrade to X1 ($19.99/mo)+.    |
| 400    | Bad request (missing or malformed `input` / `lens` / `webhook_url`). |
| 404    | No such scan (or not yours).                                 |
| 429    | Rate limited — retry with backoff.                          |

Questions: **api@thesapientcompany.com**

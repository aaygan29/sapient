# Usage-Based API Billing — Implementation Spec

**Goal:** Move Sapient to infrastructure-first, usage-based API pricing. The API is the product; the front-end is a free demo. **From the user's POV almost nothing changes** — same wallet, same "add money → run the API," same keys. What changes: (1) per-scan price drops automatically as monthly usage grows (graduated), (2) the Plans page shows 4 usage tiers instead of Pro/Studio, (3) Pro/Studio/Dev Kit are removed, (4) Sapient team = admins with unlimited scans.

**Hard rule: DO NOT BREAK existing flows** (wallet top-up, Stripe webhook credit, API-key mint, free-credit-first debit, scan submit). Ship in the phases below, each independently safe.

## Final pricing (source of truth)
| Tier | Price/scan | Monthly usage bracket |
|---|---|---|
| Pay-as-you-go | $2.50 (250¢) | 0 – 1,000 |
| Build | $1.50 (150¢) | 1,001 – 10,000 |
| Scale | $1.00 (100¢) | 10,001 – 100,000 |
| Enterprise | Custom (no price shown) | 100,000+ |

- **Graduated billing** (no cliffs): first 1k @ 250¢, next to 10k @ 150¢, next to 100k @ 100¢.
- **Free tier:** 3 free **web** scans (front-end demo, unchanged — `analysis_credits` path) **+ $10 (1000¢) one-time free API credit on signup.**
- **$10 free credit is NOT money:** lives in `sapient_monthly_credits.free_credit_remaining_cents`, never a Stripe event, excluded from revenue/ops dashboards. Debited before the paid wallet (already the behavior).
- **Scope:** graduated pricing applies to the **metered API path only** (`api/v1/*` + `chargeForScan`). The web front-end stays a free 3-scan demo. Two scan paths stay separate.
- **"1 scan" = one content item** (one video/image). A batched API request charges per item.

## Key existing files (grounded in current repo)
- `src/lib/plans.ts` — hardcoded `PLANS` array (source of truth for Billing UI). **EDIT.**
- `src/lib/sapientApiPricing.ts` — pricing constants (`PER_MODEL_PER_CALL_USD_CENTS` qualia=250, `FREE_MONTHLY_CREDIT_USD_CENTS=250`). **EDIT — add bracket engine here.** ⚠️ Notes the real metering gateway is an external deployment — see Phase 0.
- `api/_sapient_helpers.ts` — `chargeForScan()` (debits free credit → deposit), `hasBalanceForScan()`. **EDIT — apply bracket rate.**
- `api/v1/analyze.ts`, `api/v1/scans.ts` — metered endpoints. **EDIT — use bracket rate.**
- `api/v1/wallet.ts`, `api/sapient/usage.ts` — balance/usage read for display. **EXTEND.**
- `src/ApiKeysSection.tsx` — dev dashboard (keys, balance hero, MTD usage). **EXTEND — tier display.**
- `src/BrandSettings.tsx` — Billing tab, renders `PLANS.map(...)`. **EDIT — 4 tiers, remove Pro/Studio/Dev Kit.**
- `src/components/SyncProgressCard.tsx` — progress-bar reference for the "X/1,000 to next tier" UI.
- `api/sapient/deposit-checkout.ts`, `api/sapient/stripe-webhook.ts` — Stripe top-ups (KEEP working; extend for auto-refill in Phase 6).
- `supabase/migrations/` — add migrations here. Decrement RPCs: `sapient_decrement_monthly_credit`, `sapient_decrement_deposit`.
- Conventions: Tailwind v4 (`src/index.css` `@theme`, `.card`/`.btn-primary`), lucide-react, custom components with `isDarkMode` className strings. **Match these — no shadcn/Radix.**

---

## Phase 0 — Investigate the external metering gateway (DO FIRST)
`sapientApiPricing.ts` states real per-call enforcement runs in a separate deployment not in this repo. Before building enforcement:
- Trace how that gateway obtains pricing (does it import this repo's config, call an endpoint, or hardcode?).
- Document whether the bracket engine must be mirrored there, and how.
- **Output a short findings note** appended to this file. Do not consider Phase 2 "done" until enforcement is confirmed end-to-end (in-repo charge path AND the gateway).

## Phase 1 — Pricing config / source of truth (no behavior change yet)
1. In `sapientApiPricing.ts`: add pure `bracketRateCentsForUsage(monthlyScanCount: number): number` returning 250/150/100. Add `BRACKETS` constant + `tierForUsage()` → `{ key, label, rateCents, nextThreshold }`.
2. Change free grant: `$10` one-time on signup (1000¢). Keep `free_credit_remaining_cents` as the store; stop monthly replenish (or set monthly free to 0). Flag clearly as non-revenue.
3. In `plans.ts`: replace `PLANS` with the 4 usage tiers (informational display objects: label, price/scan string, bracket range, CTA). Remove Pro/Studio/Dev Kit. Keep `tierHasApiAccess`/helpers compiling.

## Phase 2 — Rate engine in the charge path
1. In `api/_sapient_helpers.ts` `chargeForScan()` (and `hasBalanceForScan` precheck): compute the org's **current calendar-month scan count** from `sapient_usage_events` (UTC month), pass through `bracketRateCentsForUsage()`, and write that as `cost_usd_cents`. Keep idempotency + free-credit-first → deposit order intact.
2. Apply the same in `api/v1/analyze.ts` / `api/v1/scans.ts` where price is resolved.
3. Mirror the bracket logic to the external gateway per Phase 0.
4. Edge cases: a single request that crosses a bracket boundary charges each scan at its own bracket; month rolls over at UTC 1st.

## Phase 3 — Tier + usage display (the visible add)
Extend `api/sapient/usage.ts` (or `v1/wallet.ts`) to return: current tier, scans this month, current rate, next threshold. In `src/ApiKeysSection.tsx`, add to the balance hero: **current plan badge** (Pay-as-you-go / Build / Scale), `X / 1,000 scans this month`, current $/scan, and a **progress bar to the next (cheaper) tier** (match `SyncProgressCard.tsx`). Wallet/top-up UI otherwise unchanged.

## Phase 4 — Plans page rewrite
In `src/BrandSettings.tsx` Billing tab: render the 4 usage tiers from `plans.ts` (informational, not selectable — Enterprise = "Talk to us", no price). Remove the Pro/Studio cards and the "Dev Kit" strip. **Keep the wallet summary, top-up button, and API-key section exactly as-is.** Free tier line: "3 free web scans + $10 API credit on signup."

## Phase 5 — Remove Studio/Pro + Sapient-team admins (migration)
No real paying customers exist — no grandfathering needed.
1. Migration in `supabase/migrations/`: set all `organizations.plan` off Pro/Studio (to the usage-based default); clear stale subscription `credit_overrides` except admins.
2. **Sapient team = admin + unlimited:** identify by `@thesapientcompany.com` email / the Sapient org; grant an admin/unlimited flag (e.g. `credit_overrides` with effectively-unlimited `max_credits`, or an `is_admin`/`unlimited` flag respected by `hasBalanceForScan`/`chargeForScan` to bypass debit). Verify these accounts never get charged.

## Phase 6 — Auto-refill execution (net-new; build last)
Currently only `sapient_deposits.auto_refill_threshold_cents` exists (read-only). Add:
1. Save card on first top-up via Stripe **SetupIntent**; store `auto_refill_enabled` + `auto_refill_amount_cents` + payment-method id.
2. After a debit in `chargeForScan`, if `balance < threshold` and enabled, charge the saved card via **PaymentIntent** (off-session) and credit the deposit (reuse the webhook credit logic). Idempotent; handle failures gracefully (notify, don't hard-block until truly empty).
3. UI toggle in `ApiKeysSection.tsx` top-up area.

## Phase 7 — Verify (don't break anything)
Confirm still working: wallet top-up + webhook credit, API-key mint/verify, scan submit on both paths, free-credit-first debit, balance display. Test bracket transitions (cross 1k/10k), month rollover, admin bypass, insufficient-balance block. Open a PR off `main`.

---

## Phase 0 Findings — the external metering gateway (investigated 2026-06-17)

**TL;DR: For the documented `www.thesapientcompany.com/api/v1/*` endpoints, the LIVE meter is IN THIS REPO. The external `sapient-api-v2/gateway` is NOT the live charge path for them, so the bracket engine is enforced end-to-end in-repo. The external gateway only needs mirroring IF traffic is ever routed through `api.thesapientcompany.com` directly (a key minted here also authenticates there).**

### How pricing/charging actually flows today
1. The two public endpoints — `api/v1/analyze.ts` and `api/v1/scans.ts` — authenticate the bearer (`verifyApiKey`), precheck balance (`hasBalanceForScan`), submit the run via `dispatchMaryRun` (in-process), and the **debit happens in `chargeForScan` (api/_sapient_helpers.ts)**:
   - inline in `analyze.ts` for synchronous completions, and
   - in `settleApiRun()` (api/mary-run.ts ~line 1045) on the poll/completion path for async runs.
2. `chargeForScan` is the ONE place that writes `cost_usd_cents` onto `sapient_usage_events` and debits free credit → deposit (via RPCs `sapient_decrement_monthly_credit` then `sapient_decrement_deposit`). It is idempotent (anchored on `request_id = 'analyze:'+runId`, PK 23505 = already charged).
3. Crucially, both call sites invoke `chargeForScan(orgId, keyId, runId)` **without passing `cents`** — so the price is resolved INSIDE `chargeForScan`. That makes it the correct single chokepoint for the bracket engine: change it once and both endpoints + sync/async paths bill graduated automatically.
4. `src/lib/sapientApiPricing.ts` is display + in-repo source of truth. Its header comment historically warned that a separate gateway is the meter, but its OWN closing note (and the code) confirm: *"A separate `sapient-api-v2/gateway` repo exists but is NOT the live meter for the documented www.thesapientcompany.com/api/v1 endpoints."*

### Does the gateway import this repo, call an endpoint, or hardcode?
- The gateway repo is **not present on this machine** (`~/Desktop/sapient-api*` / `gateway*` — none found). Based on the in-repo documentation it maintains its **own hardcoded pricing table** (the `PRICE_CHANGE_CHECKLIST` says "Mirror them in the gateway pricing table (sapient-api-v2/gateway)"). It does not import this repo's TS at runtime.
- Therefore: **flat-rate values would need manual mirroring; the GRADUATED bracket logic would also need to be re-implemented there** — BUT only for traffic that hits `api.thesapientcompany.com` directly. The product's documented endpoints do not, so end-to-end enforcement of brackets is satisfied by this repo alone.

### What must be mirrored externally (only if the gateway becomes a live meter)
Implemented in this repo (authoritative): `bracketRateCentsForUsage()`, `BRACKETS`, `tierForUsage()` in `src/lib/sapientApiPricing.ts`; consumed by `chargeForScan` + `hasBalanceForScan`. To mirror in `sapient-api-v2/gateway` you must replicate, against the SAME `sapient_usage_events` / `sapient_monthly_credits` / `sapient_deposits` tables and the SAME RPCs:
- the bracket thresholds + rates: **250¢ (0–1,000), 150¢ (1,001–10,000), 100¢ (10,001–100,000), custom (100,000+)**, computed per **calendar-month (UTC) scan count** for the org;
- **graduated** semantics: a request crossing a bracket boundary charges each scan at its own bracket rate (the engine here exposes `bracketRateCentsForUsage(n)` per-item; sum per item for a batch);
- the **$10 (1000¢) one-time signup free credit** stored in `sapient_monthly_credits.free_credit_remaining_cents`, debited before the deposit (never a Stripe event, excluded from revenue);
- the **admin/unlimited bypass** for `@thesapientcompany.com` / the Sapient org (no debit).

Net: no blocking dependency on the external repo for the documented product. Flagged for the user in the final report.

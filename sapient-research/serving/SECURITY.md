# Security review — Sapient-1 serving layer

Two stated goals drive this design:

1. **Don't let people reverse-engineer the model.**
2. **Don't let confidential data leak** (across clients or out of the system).

This doc maps each goal to concrete controls, marks what's implemented vs. what
must be hardened before production, and is honest about residual risk.

---

## Goal 1 — resist reverse-engineering

| Control | Status | Notes |
|---|---|---|
| **Weights never shipped** | ✅ in design | The model runs only server-side; the client package (`sapient-client`) is pure HTTP. No SDK contains weights. |
| **API-only access** | ✅ | Clients see inputs→outputs, never the model. |
| **Coarse outputs** | ✅ | Returns ROI + parcel summary + image. The full 1000-dim parcel vector is withheld unless a client is explicitly `allow_full_parcels`. Raw 20,484-vertex maps are never exposed. |
| **Per-client rate limit + daily quota** | ✅ (single-instance) | Caps how many (input,output) pairs anyone can harvest for distillation. `ratelimit.py`. **Swap to Redis for multi-instance.** |
| **Extraction-pattern monitoring** | ⛔ TODO | Add anomaly detection (volume spikes, systematic input sweeps) + alerting; auto-throttle/revoke suspicious keys. |
| **Output watermark / canaries** | ⛔ optional | Subtle, per-client perturbations to prove theft later if distilled. |
| **Terms of use** | ⛔ legal | Contract forbidding reverse-engineering / training on outputs. Belt-and-suspenders with the technical controls. |

**Residual risk (honest):** a hosted API cannot make extraction *impossible* — a
determined, paying client can still attempt to distill a surrogate from outputs.
The controls make that **slow, expensive, and detectable**, which is the realistic
goal. Lower output granularity + tighter quotas raise the cost further.

---

## Goal 2 — protect confidential data

| Control | Status | Notes |
|---|---|---|
| **TLS in transit** | ✅ (at edge) | Terminated by Modal/the ingress. Don't run plaintext. HSTS header set. |
| **Default-deny auth** | ✅ | Every `/v1` route requires a valid key; unknown/missing → 401. `auth.py`. |
| **Keys stored hashed** | ✅ | Only SHA-256 hashes on disk (`keys.json`, 0600). Raw key shown once at issue. |
| **Per-client identity** | ✅ | Each client has its own key → revoke/rotate/attribute per client. |
| **Tenant isolation (no IDOR)** | ✅ | `GET /v1/jobs/{id}` returns a result only to its owner; otherwise an identical 404 (no existence leak). UUID4 job IDs (unguessable). `storage/jobs.py`. |
| **Data minimization / ephemerality** | ✅ | Raw uploads are read into memory, used, then dropped — never persisted. Only the coarse result is stored, with a short TTL (`SAPIENT_RESULT_TTL`), purged on access. |
| **Upload validation** | ✅ | Content-type checked (`video/*`,`audio/*`); size capped (`SAPIENT_MAX_UPLOAD_MB`) by reading with a hard limit. |
| **No payloads/PII in logs** | ✅ by design | Errors surface a type+message, not input contents. Keep it that way; don't add request-body logging. |
| **Secrets management** | ✅ pattern | HF token + key store via Modal secrets / env, never in code. `.env`/`keys.json` are gitignored. |
| **Security headers** | ✅ | `nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, HSTS. |
| **Docs surface hidden** | ✅ | Swagger/OpenAPI off unless `SAPIENT_DOCS=1`. |

---

## Dev → production checklist (do NOT skip before real clients)

- [ ] **Move rate-limit + job store to Redis / object store** — the in-memory
      versions are correct for ONE instance only; with autoscaling they don't share state.
- [ ] **Real malware/format scanning** of uploads (beyond content-type + size);
      guard against decompression bombs and malformed-media crashes in the decoders.
- [ ] **Run inference off the event loop** — real GPU work must be offloaded
      (Modal function / executor), not run inline, or it blocks the server.
- [ ] **Audit logging** of auth events + per-client usage (for the extraction monitor).
- [ ] **Key rotation** procedure + expiry; revoke on offboarding.
- [ ] **Rate-limit the unauthenticated surface** (`/healthz`, `/playground`) at the edge to blunt DoS.
- [ ] **Pin + scan dependencies**; run `ecc-security-review` against this tree.
- [ ] **Confirm provenance copy** in `engine/provenance.py` with legal before launch.

---

## Faithfulness note (integrity is a security property here)

The API must never present Sapient-1 as the papers' "Mary" model or attach the
papers' metrics to it. `GET /v1/info` discloses: served model, `is_paper_model:false`,
architecture basis, CC0 training data, and that per-ROI values are **predicted
activation, not correlation-with-measured-fMRI**. Misrepresentation here is both
an integrity failure and a due-diligence risk.

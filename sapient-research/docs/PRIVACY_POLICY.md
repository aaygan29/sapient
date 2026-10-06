# Privacy Policy — INTERNAL DRAFT

> **Status: internal draft. Not legal advice and not customer-facing.** This is a basic
> template for internal review. Have counsel review before publishing anything from it.
> Do not link this file from the product; it is excluded from the deployable build
> (`.vercelignore`).

**Company:** The Sapient Company ("Sapient", "we", "us")
**Last updated:** 2026-08-15 (draft)

## 1. Scope
This policy describes how Sapient handles information in connection with our brand-intelligence
and creative-analysis products. It covers customer account data and the creative assets
(video, audio, text, images) customers submit for analysis.

## 2. Information we collect
- **Account data:** name, email, organization, authentication identifiers (via Clerk).
- **Submitted content:** ad creative and campaign material a customer uploads for scoring.
- **Usage data:** product analytics and error/performance telemetry (PostHog, Sentry).
- **Billing data:** processed by Stripe; we do not store full card numbers.

## 3. What we do NOT collect
- We do **not** collect neural, biometric, or health data from end viewers. Sapient's scores
  are **model predictions from creative content**, not measurements of any real person's brain
  or body. No individual is scanned, tracked, or profiled.
- We do not sell personal information.

## 4. How we use information
- To provide and improve creative-analysis features.
- To operate accounts, billing, security, and support.
- Aggregate, de-identified analysis of usage to improve models. Customer creative is not used
  to train shared models without explicit permission.

## 5. Sub-processors
Clerk (auth), Supabase (database), Stripe (payments), Vercel (hosting), Resend (email),
PostHog (analytics), Sentry (errors), and AI providers (Anthropic, OpenAI, xAI) for analysis.
Each processes data only as needed to provide their service.

## 6. Data retention & deletion
Customer content is retained for the life of the account and deleted on request or within a
defined window after account closure. Contact us to request access, correction, or deletion.

## 7. Security
Encryption in transit (HSTS enforced), access controls, Row-Level Security on the database,
and secret management. No system is perfectly secure; we work to protect data reasonably.

## 8. Data-subject rights
Depending on jurisdiction (e.g. GDPR/CCPA), individuals may have rights to access, correct,
delete, or port their data, and to object to certain processing. Requests: privacy@thesapientcompany.com.

## 9. Changes
We will update this policy as the product and law evolve, and note the revision date.

## 10. Contact
privacy@thesapientcompany.com

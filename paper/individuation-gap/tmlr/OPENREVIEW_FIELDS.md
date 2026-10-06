# TMLR OpenReview submission — copy-paste fields

Submit at: https://openreview.net/group?id=TMLR (click "Submit").
Upload the PDF `tmlr/main.pdf` and, as Supplementary Material, `tmlr/supplementary.zip`.
TMLR is an **archival, peer-reviewed journal**. Its originality rule forbids overlap with any
published, accepted, or **parallel-submitted** archival work; overlap with non-archival workshops
or arXiv preprints is allowed. Before submitting, confirm nothing overlapping is under review
elsewhere.

---

## Title
Who, Not What: Identity Is Recoverable but Individual Behaviour Is Not in Non-Invasive and In-Silico Neuroforecasting

## Abstract
A growing set of products and clinical tools claims to read an individual's mental state or behaviour from brain and body signals, increasingly from a predicted brain rather than a measured one: a stimulus-to-cortex encoder maps media or context to cortical activity, which is then read as if it had been recorded. We show that forecasting from a predicted brain obeys a simple structural limit, the individuation gap. Forecasting value separates into a population part and an individual part. The population part improves with aggregation, which we prove, whereas the individual part depends on between-subject variance that a zero-shot encoder does not carry, so it is zero without per-subject enrollment. Three independent datasets show the consequence. Non-invasive readouts recover who a person is, with in-scanner fingerprinting at 100% and a behavioural fingerprint at 14 times chance, but not what they will do, since the direct link from an individual's neural signal to that individual's behaviour abstains under a calibrated test. A Bayes-optimal forecaster inherits the same limit, adding no individual value at zero-shot and only a modest, saturating gain once per-subject data is paid for. Combining signals does not remove the limit, because each added modality carries its own noise. These results agree with the measurement-reliability literature, in which individual brain-behaviour effects are small and require large samples, and they motivate a read-out that reports a value only when it has earned the label. The contribution is a precise and verifiable account of what predicted-brain readouts can and cannot deliver at the level of the individual.

## Authors
Aayush Gandhi (your OpenReview profile). Keep the PDF anonymized; the author field on OpenReview
is hidden from reviewers.

## Submission Type
Regular submission (main content is under 12 pages; references and appendix do not count).

## Competing Interests
(Recommended, visible only to editors.) "An author was employed, within the past 36 months, by a
company developing in-silico neuroforecasting / brain-based audience-measurement products, a
product area this paper evaluates. No other competing interests."
(If you prefer not to disclose the employer by category, at minimum do not enter "N/A" here given
the subject matter; consult the Ethics Guidelines. Disclosure here does not affect anonymity.)

## Human Subjects Reporting
"N/A. The study uses only existing, publicly available, de-identified datasets (NSD, BOLD5000,
NARPS ds001734, ds000005) and collects no new human-subjects data; no IRB approval was required."

## Broader Impact Statement
Included in the manuscript (end of Discussion). Flagged because the work bears on neurotechnology
claims about individuals.

## License
CC BY 4.0 (TMLR default and required).

## Suggested Action Editors
Pick 1–3 Action Editors from the TMLR AE list whose expertise covers computational models of
natural learning at the behavioural/neural level, evaluation methodology / reliability, or
trustworthy ML. (The five Editors-in-Chief are Laurent Charlin, Gautam Kamath, Yoshitomo
Matsubara, Naila Murray, and Nihar B. Shah; do not list EiCs as AEs.) Browse AEs on the TMLR
OpenReview group and choose by scope match before submitting.

## Changes Since Last Submission / Previous TMLR Submission
N/A (first submission).

---

## Files to upload
- PDF: `tmlr/main.pdf`
- Supplementary (ZIP): `tmlr/supplementary.zip` (Lean proofs + simulation/figure code)
- Style files used (already applied): `tmlr/tmlr.sty`, `tmlr/fancyhdr.sty`, `tmlr/tmlr.bst`
  (official, from https://jmlr.org/tmlr and the JmlrOrg/tmlr-style-file repository)

## Pre-submission checklist
- [ ] OpenReview profile complete (affiliations, conflicts, history) — new profiles can take up to two weeks to activate.
- [ ] Confirm no overlapping result is under review at another archival venue.
- [ ] Regenerate the real-data numbers from a fixed commit and confirm they match the PDF.
- [ ] Verify the PDF has no identifying text (it is built anonymized).
- [ ] Check your TMLR submission quota for the year.

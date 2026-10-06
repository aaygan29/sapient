# Submission guide

## Recommended venue: TMLR (OpenReview)

**Transactions on Machine Learning Research** — https://jmlr.org/tmlr/ — submitted through
OpenReview at https://openreview.net/group?id=TMLR.

Why TMLR fits this paper:
- **Rolling submission, no deadline.** All relevant conference/workshop deadlines for this cycle
  (NeurIPS 2026 workshops, ICLR 2027) have passed; TMLR can be submitted today.
- **Acceptance criterion is correctness and support, not novelty or impact.** The two TMLR
  questions are: are the claims correct and well supported, and would some audience be interested.
  A rigorous result with honest negative findings is exactly what this rewards.
- **OpenReview-hosted**, as requested; reviews are public and constructive.
- No page limit, single-column, so the current format fits with minimal change.

### Steps
1. Create or confirm an OpenReview profile (can take up to two weeks for a new one, so start now).
2. Replace the local preamble with the official TMLR style: download `tmlr.sty` and
   `fancyhdr.sty` from the TMLR author kit (https://jmlr.org/tmlr/author-guide.html) and
   `\usepackage{tmlr}`. The body, figures, theorems, and references carry over unchanged.
3. TMLR is single-blind by default for the camera-ready and double-blind during review; submit the
   anonymized build (no author block) during review.
4. Suggested primary area: "Social and ethical aspects / evaluation methodology" or
   "Neuroscience and cognitively inspired AI."

## Journal alternative: PLOS ONE

https://journals.plos.org/plosone/ — rolling, open access, judged on methodological soundness
rather than impact. Good fit for the same reasons. Reformat to the PLOS LaTeX template
(single-column) and add the required sections (Methods, Data Availability). APC applies; fee
waivers are available on request.

Second journal option: **Imaging Neuroscience** (MIT Press, rolling, open access, neuroimaging
methods audience).

## Before submitting (checklist)
- [ ] Regenerate all real-data numbers from a frozen commit and confirm they match the paper.
- [ ] Confirm no overlapping result is concurrently under review at another venue.
- [ ] Run a final pass for anonymization if the venue requires it during review.
- [ ] Rebuild figures from `code/` and the paper from `main.tex`.

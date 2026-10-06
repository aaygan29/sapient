import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic

/-!
# The individuation gap, formally

This file recomputes the neuroforecasting kernel around the central claim of the
paper: forecasting value from a predicted brain splits into a population term and
an individual term, and the two are provably asymmetric.

Model. A target `Y = S + ε` has a predictable signal component `S` with
*identifiable* variance `a ≥ 0` (the between-unit variance the predictor can
actually resolve) and idiosyncratic noise of variance `v > 0`. The best-case
squared correlation between the predictor and the target is

    ρ²(a, v) = a / (a + v).

Two regimes set `a`:

* **Population / enrolled.** Aggregating over `N` units shrinks the noise to
  `v / N`, and the identifiable signal variance `a > 0` is retained. The squared
  correlation `ρ²(a, v / N)` strictly increases with `N` (`aggregation_improves`).

* **Individual / unseen (zero-shot).** A zero-shot encoder emits the same
  population feature for every unseen unit, so the predictor has *no* between-unit
  variance: its identifiable signal variance is `a = 0`. Then `ρ²(0, v) = 0`:
  there is no individual forecasting signal to recover, for any noise level and
  any sample size (`unseen_individual_is_zero`, `unseen_individual_zero_any_N`).

The gap theorem states these together: the unseen-individual correlation is
strictly below the population correlation whenever the population has any
identifiable signal. The arithmetic is elementary; the point is that the
impossibility of zero-shot individuation is a theorem about the estimand, not a
limitation of a particular model or dataset.
-/

namespace IndividuationGap

/-- Best-case squared correlation between a predictor with identifiable signal
variance `a` and a target `Y = S + ε` with idiosyncratic-noise variance `v`. -/
noncomputable def rhoSq (a v : ℝ) : ℝ := a / (a + v)

/-- A squared correlation never reaches 1 while idiosyncratic noise remains. -/
lemma rhoSq_lt_one {a v : ℝ} (ha : 0 < a) (hv : 0 < v) : rhoSq a v < 1 := by
  unfold rhoSq
  rw [div_lt_one (by positivity)]
  linarith

/-- Less idiosyncratic noise means a higher (squared) correlation. -/
lemma rhoSq_antitone {a v₁ v₂ : ℝ} (ha : 0 < a) (hv₁ : 0 ≤ v₁) (h : v₁ < v₂) :
    rhoSq a v₂ < rhoSq a v₁ := by
  unfold rhoSq
  gcongr

/-- **Population term: aggregation strictly improves prediction.** For any real
group size `N > 1`, the signal correlates more strongly with the `N`-averaged
outcome than with a single individual's outcome. -/
theorem aggregation_improves {a v N : ℝ} (ha : 0 < a) (hv : 0 < v) (hN : 1 < N) :
    rhoSq a v < rhoSq a (v / N) := by
  have hN0 : 0 < N := by linarith
  have hlt : v / N < v := by
    have h2 : v / N < v / 1 := by gcongr
    simpa using h2
  exact rhoSq_antitone ha (le_of_lt (div_pos hv hN0)) hlt

/-- **Individual term under a zero-shot encoder: no identifiable signal.** With no
between-unit variance (`a = 0`), the squared correlation is exactly zero,
regardless of the idiosyncratic-noise level. -/
theorem unseen_individual_is_zero (v : ℝ) : rhoSq 0 v = 0 := by
  unfold rhoSq
  simp

/-- The zero-shot individual term stays zero no matter how much data is pooled:
aggregation cannot manufacture an individual signal that was never in the
predictor. -/
theorem unseen_individual_zero_any_N (v N : ℝ) : rhoSq 0 (v / N) = 0 := by
  unfold rhoSq
  simp

/-- **The individuation gap.** Whenever the population has any identifiable signal
(`a > 0`) and there is noise (`v > 0`), the population correlation at any sample
size `N > 0` is strictly greater than the zero-shot individual correlation, which
is zero. The two estimands are provably separated. -/
theorem individuation_gap {a v N : ℝ} (ha : 0 < a) (hv : 0 < v) (hN : 0 < N) :
    rhoSq 0 (v / N) < rhoSq a (v / N) := by
  rw [unseen_individual_zero_any_N]
  unfold rhoSq
  have hvN : 0 < v / N := div_pos hv hN
  positivity

end IndividuationGap

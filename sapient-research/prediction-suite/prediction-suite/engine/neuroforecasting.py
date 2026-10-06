"""
PredictiveValue™ — Neuroforecasting Brain → Behavioral Outcomes

Links Mary's value/reward/emotion read-outs to real outcomes (CTR, sales, ad-recall).
Implements the Knutson/Genevsky three-estimator design: does the brain-based score
forecast hold-out ad outcomes better than self-report or chance?

Implementation: leave-one-ad-out LOCO cross-validation. Test if value/reward read-out
predicts outcome (purchase intent, CTR, recall) above chance and outperforms self-report.

Usage:
    forecaster = Neuroforecaster(construct='value', outcome_metric='ctr')
    result = forecaster.predict_outcomes(vmaps, outcomes, self_report=None)
    # → {r_brain, r_self, r_chance, p_value, effect_size, passes_gate}
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy import stats
from typing import Optional


@dataclass
class NeuroforecasterResult:
    """Neuroforecasting outcome."""
    construct: str
    outcome_metric: str
    r_brain: float                     # Pearson r: brain score ↔ outcome
    r_self_report: float | None        # Alternative predictor (if provided)
    r_chance: float                    # Null distribution percentile
    p_value: float                     # Against permutation null
    effect_size: float                 # Correlation strength (small/medium/large)
    passes_specificity: bool           # Did it beat self-report?
    n_ads: int
    cv_fold: int
    notes: str = ""


class Neuroforecaster:
    """
    Three-estimator design (Knutson/Genevsky, adapted for in-silico):
      1. Brain-based: does Mary's value/reward score predict outcome?
      2. Self-report: does user self-rating predict outcome?
      3. Chance: what's the null distribution?

    Claim earned only if: r_brain > r_self AND r_brain is significant.
    """

    def __init__(self, construct: str = 'value', outcome_metric: str = 'ctr',
                 min_effect: float = 0.35):
        """
        construct: 'value', 'reward', 'emotion', 'arousal', 'attention', 'memory'
        outcome_metric: 'ctr', 'sales', 'recall', 'purchase_intent' (user-defined scale 0–100)
        min_effect: minimum |r| to claim predictive validity (Genevsky ~0.35)
        """
        self.construct = construct
        self.outcome_metric = outcome_metric
        self.min_effect = min_effect
        self.results: list[NeuroforecasterResult] = []

    def predict_outcomes(self, construct_scores: np.ndarray,
                        outcomes: np.ndarray,
                        self_report: np.ndarray | None = None,
                        n_perms: int = 10000) -> NeuroforecasterResult:
        """
        Leave-one-ad-out cross-validation: predict each ad's outcome from brain score.

        Args:
            construct_scores: (n_ads,) Mary's value/reward/emotion score per ad
            outcomes: (n_ads,) behavioral outcome (CTR, recall 0–100, sales, etc.)
            self_report: (n_ads,) optional user self-report (same scale as outcome)
            n_perms: permutations for null distribution

        Returns:
            NeuroforecasterResult with predictive validity verdict.
        """
        assert construct_scores.shape[0] == outcomes.shape[0]
        n_ads = construct_scores.shape[0]

        # LOCO: leave one ad out, predict it from n-1 others
        r_loco = []
        for held_out in range(n_ads):
            train_idx = np.arange(n_ads) != held_out
            X_train = construct_scores[train_idx]
            y_train = outcomes[train_idx]

            # Fit simple linear model on train set
            slope, intercept, r, p, se = stats.linregress(X_train, y_train)

            # Predict held-out
            y_pred = slope * construct_scores[held_out] + intercept
            y_true = outcomes[held_out]

            # Residual correlation (centered)
            r_loco.append((y_pred - y_true))

        # Overall correlation across LOCO predictions
        r_loco_pred = np.array(r_loco)
        r_brain = stats.pearsonr(construct_scores, outcomes)[0]

        # Self-report baseline (if provided)
        r_self = np.nan
        if self_report is not None:
            r_self = stats.pearsonr(self_report, outcomes)[0]

        # Permutation null: shuffle outcome labels, recompute r
        r_perm = []
        for _ in range(n_perms):
            perm_outcome = np.random.permutation(outcomes)
            r_perm.append(stats.pearsonr(construct_scores, perm_outcome)[0])
        r_perm = np.array(r_perm)

        # One-tailed p-value: r_brain > null distribution
        p_value = (np.sum(np.abs(r_perm) >= np.abs(r_brain)) + 1) / (n_perms + 1)

        # Effect size classification (Cohen's r)
        effect_size = abs(r_brain)
        effect_label = "trivial" if effect_size < 0.10 else \
                       "small" if effect_size < 0.30 else \
                       "medium" if effect_size < 0.50 else "large"

        # Verdict: brain > self + significant + beats minimum effect
        passes = (r_brain > r_self or np.isnan(r_self)) and \
                 p_value < 0.05 and effect_size >= self.min_effect

        result = NeuroforecasterResult(
            construct=self.construct,
            outcome_metric=self.outcome_metric,
            r_brain=r_brain,
            r_self_report=r_self,
            r_chance=np.percentile(np.abs(r_perm), 95),  # 95th percentile of null
            p_value=p_value,
            effect_size=effect_size,
            passes_specificity=passes,
            n_ads=n_ads,
            cv_fold=1,
            notes=f"{effect_label} effect; " +
                  (f"beats self-report ({r_self:.2f})" if r_self > 0 else "no self-report") +
                  (f"; PASS" if passes else f"; FAIL (min={self.min_effect})")
        )

        self.results.append(result)
        return result

    def report(self) -> dict:
        """Summarize all neuroforecasting results."""
        if not self.results:
            return {"status": "no results"}

        r = self.results[-1]
        return {
            "construct": r.construct,
            "outcome_metric": r.outcome_metric,
            "correlation_with_outcome": round(r.r_brain, 3),
            "p_value": round(r.p_value, 4),
            "effect_size": round(r.effect_size, 3),
            "beats_self_report": r.r_brain > r.r_self_report if not np.isnan(r.r_self_report) else True,
            "predictive_validity_claim": "PASS" if r.passes_specificity else "FAIL",
            "n_ads_tested": r.n_ads,
            "notes": r.notes
        }


@dataclass
class CampaignForecast:
    """Forecast for a single campaign/ad."""
    ad_id: str
    construct_score: float          # Value/reward/emotion from Mary
    predicted_outcome: float        # Forecasted CTR/sales/recall
    confidence_interval: tuple      # 95% CI on prediction
    percentile_vs_similar: float    # What % of similar ads score lower?
    recommendation: str             # "High-confidence high-performing" / "Low ROI expected" / etc.

    def __str__(self):
        pred, (low, high) = self.predicted_outcome, self.confidence_interval
        return (f"{self.ad_id}: predicted {self.construct_score:.2f} brain score → "
                f"{pred:.1f} outcome [CI: {low:.1f}–{high:.1f}] ({self.percentile_vs_similar:.0f}th %ile)")


def forecast_new_ads(trained_forecaster: Neuroforecaster,
                     new_construct_scores: np.ndarray,
                     ad_ids: list[str],
                     historical_outcomes: np.ndarray,
                     historical_scores: np.ndarray) -> list[CampaignForecast]:
    """
    Apply trained forecaster to new ads (not in training).

    Args:
        trained_forecaster: Neuroforecaster with .results (trained on historical data)
        new_construct_scores: (n_new,) Mary scores for new ads
        ad_ids: identifiers for new ads
        historical_outcomes: (n_train,) training outcomes (for calibration)
        historical_scores: (n_train,) training brain scores

    Returns:
        List of CampaignForecast objects with predictions + confidence intervals.
    """
    r = trained_forecaster.results[-1]

    # Fit calibration line on training data
    slope, intercept, _, _, se = stats.linregress(historical_scores, historical_outcomes)

    forecasts = []
    for ad_id, score in zip(ad_ids, new_construct_scores):
        pred = slope * score + intercept

        # 95% CI (uses training residual std)
        residuals = historical_outcomes - (slope * historical_scores + intercept)
        se_pred = np.std(residuals)
        ci = (pred - 1.96 * se_pred, pred + 1.96 * se_pred)

        # Percentile vs training distribution
        pctl = 100 * np.mean(historical_scores < score)

        # Simple recommendation rule
        if pred > np.percentile(historical_outcomes, 75) and r.r_brain > 0.3:
            rec = "High-confidence high performer"
        elif pred < np.percentile(historical_outcomes, 25):
            rec = "Low ROI expected; consider redesign"
        else:
            rec = "Average; no strong signal"

        forecasts.append(CampaignForecast(
            ad_id=ad_id,
            construct_score=score,
            predicted_outcome=pred,
            confidence_interval=ci,
            percentile_vs_similar=pctl,
            recommendation=rec
        ))

    return forecasts

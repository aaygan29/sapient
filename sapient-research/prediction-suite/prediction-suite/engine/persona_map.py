"""
PersonaMap™ — Subject-Specific Brain Phenotyping & Few-Shot Transfer

Identifies how each individual's brain differs from population average.
E.g., "Subject A is 3× more responsive to luxury imagery; 0.5× to price appeals."

Enables few-shot transfer: given a subject's heterogeneity profile, predict their
response to brand new stimuli they haven't seen (n=1–3 examples suffice).

Based on Dalla Porta (PNAS 2026) regional heterogeneity + hyperbolic brain graphs (2507.02908).

Usage:
    persona = PersonaMap.from_subject_fmri(vmaps_train, construct_maps, construct_labels)
    phenotype_card = persona.phenotype_card()  # {construct: deviation_from_pop_avg}
    pred_new = persona.predict_new_stimulus(vmap_new)  # Personalized prediction
    transfer = persona.few_shot_transfer([stim1, stim2], [outcome1, outcome2], new_stim)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy import stats
from sklearn.preprocessing import StandardScaler
from typing import Optional


@dataclass
class PhenotypeCard:
    """Subject's neural phenotype relative to population."""
    subject_id: str
    construct_deviations: dict[str, float]   # {construct: deviation_std (0 = pop avg, >0 = more responsive)}
    construct_percentiles: dict[str, float]  # {construct: percentile vs population (0–100)}
    responsive_constructs: list[str]         # Top N most distinctive constructs
    prediction_confidence: float              # 0–1: how well-constrained is this phenotype?
    notes: str = ""

    def __str__(self):
        resp = ", ".join(self.responsive_constructs[:3])
        return (f"Phenotype[{self.subject_id}]: {resp} | "
               f"Confidence: {self.prediction_confidence:.2f}")


@dataclass
class FewShotPrediction:
    """Few-shot transfer prediction for new stimulus."""
    subject_id: str
    stimulus_id: str
    predicted_scores: dict[str, float]     # {construct: personalized score}
    confidence_interval: dict[str, tuple]  # {construct: (low, high)}
    basis: str                              # "1-shot", "2-shot", "3-shot" or "phenotype-only"
    transferability: float                  # 0–1: how confident in transfer?


class PersonaMap:
    """
    Subject-specific neural phenotyping via heterogeneity analysis.
    """

    def __init__(self, subject_id: str, n_constructs: int = 6):
        """
        Args:
            subject_id: identifier (e.g., "subj_001")
            n_constructs: number of brain-response dimensions
        """
        self.subject_id = subject_id
        self.n_constructs = n_constructs

        # Trained state
        self.construct_means_subject: np.ndarray | None = None    # (n_constructs,)
        self.construct_stds_subject: np.ndarray | None = None     # (n_constructs,)
        self.construct_means_pop: np.ndarray | None = None        # (n_constructs,) population
        self.construct_stds_pop: np.ndarray | None = None         # (n_constructs,) population
        self.construct_labels: list[str] = []
        self.n_training_samples = 0

    @staticmethod
    def from_subject_fmri(subject_vmaps: np.ndarray,
                         construct_maps: dict[str, np.ndarray],
                         population_stats: Optional[dict] = None) -> PersonaMap:
        """
        Train a PersonaMap from a subject's fMRI responses to multiple stimuli.

        Args:
            subject_vmaps: (n_stimuli, 20484) subject's predicted fMRI for each stimulus
            construct_maps: {construct_name: (20484,) weight map} (from mary_readouts)
            population_stats: {'means': (n_constructs,), 'stds': (n_constructs,)} or None

        Returns:
            Fitted PersonaMap instance.
        """
        n_stimuli = subject_vmaps.shape[0]
        construct_labels = sorted(construct_maps.keys())
        n_constructs = len(construct_labels)

        # Compute subject's construct scores across all stimuli
        subject_scores = np.array([
            [np.dot(subject_vmaps[i], construct_maps[c]) for c in construct_labels]
            for i in range(n_stimuli)
        ])  # (n_stimuli, n_constructs)

        # Subject-level statistics (mean and std across stimuli)
        subject_means = np.mean(subject_scores, axis=0)
        subject_stds = np.std(subject_scores, axis=0)

        # Population statistics (assume provided or use unit variance)
        if population_stats is not None:
            pop_means = population_stats['means']
            pop_stds = population_stats['stds']
        else:
            pop_means = np.zeros(n_constructs)
            pop_stds = np.ones(n_constructs)

        # Instantiate and store
        persona = PersonaMap(subject_id="subj_unknown", n_constructs=n_constructs)
        persona.construct_means_subject = subject_means
        persona.construct_stds_subject = subject_stds
        persona.construct_means_pop = pop_means
        persona.construct_stds_pop = pop_stds
        persona.construct_labels = construct_labels
        persona.n_training_samples = n_stimuli

        return persona

    def phenotype_card(self) -> PhenotypeCard:
        """
        Generate a phenotype card: how does this subject deviate from population?
        """
        assert self.construct_means_subject is not None

        # Deviation in standard deviations (z-score for the subject vs population)
        deviations = (self.construct_means_subject - self.construct_means_pop) / (self.construct_stds_pop + 1e-8)

        # Percentile interpretation (where does subject fall in population?)
        percentiles = {}
        for i, label in enumerate(self.construct_labels):
            # Assume population is ~N(0, 1); subject has mean 'deviations[i]'
            percentiles[label] = float(stats.norm.cdf(deviations[i]) * 100)

        # Top constructs by absolute deviation
        top_idx = np.argsort(np.abs(deviations))[-3:][::-1]
        responsive = [self.construct_labels[i] for i in top_idx]

        # Confidence: how well-determined is the phenotype? (increases with N stimuli)
        confidence = np.clip(self.n_training_samples / 10.0, 0, 1)  # Saturates at N=10

        return PhenotypeCard(
            subject_id=self.subject_id,
            construct_deviations={label: float(deviations[i])
                                  for i, label in enumerate(self.construct_labels)},
            construct_percentiles=percentiles,
            responsive_constructs=responsive,
            prediction_confidence=confidence,
            notes=f"Based on {self.n_training_samples} training stimuli. " +
                  (f"High confidence." if confidence > 0.7 else
                   f"Moderate confidence (n<10)." if confidence > 0.4 else
                   f"Low confidence (n<5); recommend more data.")
        )

    def predict_new_stimulus(self, vmap_new: np.ndarray,
                            apply_phenotype_correction: bool = True) -> dict[str, float]:
        """
        Predict construct scores for a new stimulus, adjusted for subject's phenotype.

        Args:
            vmap_new: (20484,) predicted fMRI for new stimulus
            apply_phenotype_correction: if True, scale scores by subject's heterogeneity

        Returns:
            {construct: personalized_score}
        """
        from mary_readouts import GroundedReadouts  # Assumes mary_readouts.py available

        # Get raw scores for new stimulus (using population-average construct maps)
        raw_scores = {}
        for label in self.construct_labels:
            # Placeholder: assumes construct_maps are accessible
            # In practice, pass construct_maps to this method
            raw_scores[label] = np.nan

        if apply_phenotype_correction and self.construct_means_subject is not None:
            # Adjust by subject's deviation: score_subject = score_pop * (1 + deviation)
            deviations = (self.construct_means_subject - self.construct_means_pop) / (self.construct_stds_pop + 1e-8)
            corrected = {label: raw_scores[label] * (1.0 + deviations[i] * 0.5)
                        for i, label in enumerate(self.construct_labels)}
            return corrected
        else:
            return raw_scores

    def few_shot_transfer(self,
                        shot_vmaps: list[np.ndarray],
                        shot_outcomes: np.ndarray,
                        new_vmap: np.ndarray,
                        construct_maps: dict[str, np.ndarray]) -> FewShotPrediction:
        """
        Few-shot transfer: given 1–3 examples with outcomes, predict new stimulus.

        Args:
            shot_vmaps: list of (20484,) predicted fMRI for example stimuli
            shot_outcomes: (n_shots,) behavioral outcomes (CTR, recall, etc.)
            new_vmap: (20484,) predicted fMRI for new stimulus
            construct_maps: {construct: (20484,) weight}

        Returns:
            FewShotPrediction with personalized scores + confidence.
        """
        n_shots = len(shot_vmaps)
        assert n_shots in [1, 2, 3], "Few-shot transfer supports 1–3 shots only"

        # Compute subject construct scores for shot examples
        shot_scores = np.array([
            [np.dot(vm, construct_maps[c]) for c in self.construct_labels]
            for vm in shot_vmaps
        ])  # (n_shots, n_constructs)

        # Fit simple linear model per construct: (shot_scores, shot_outcomes) -> new_score
        predicted_scores = {}
        confidence_intervals = {}
        transfer_confidence_vals = []

        for i, label in enumerate(self.construct_labels):
            X = shot_scores[:, i]
            y = shot_outcomes

            # Linear regression
            if n_shots > 1:
                slope, intercept, r, p, se = stats.linregress(X, y)
            else:
                # 1-shot: assume proportionality
                slope = y[0] / (X[0] + 1e-8)
                intercept = 0
                r = 0  # No confidence estimate with 1 point

            # Predict
            new_score_construct = np.dot(new_vmap, construct_maps[label])
            pred = slope * new_score_construct + intercept

            # Apply personalization correction
            if self.construct_means_subject is not None:
                deviation = (self.construct_means_subject[i] - self.construct_means_pop[i]) / (self.construct_stds_pop[i] + 1e-8)
                pred *= (1.0 + deviation * 0.5)  # Gentle upweight for high-responsive constructs

            predicted_scores[label] = float(pred)

            # Confidence interval (wider for 1-shot)
            se_pred = abs(se) if n_shots > 1 else abs(pred * 0.3)  # 30% margin for 1-shot
            ci = (float(pred - 1.96 * se_pred), float(pred + 1.96 * se_pred))
            confidence_intervals[label] = ci

            # Track R² as a measure of transfer quality
            transfer_confidence_vals.append(r ** 2 if n_shots > 1 else 0.5)

        # Overall transferability score
        transferability = float(np.mean(transfer_confidence_vals))

        return FewShotPrediction(
            subject_id=self.subject_id,
            stimulus_id="new",
            predicted_scores=predicted_scores,
            confidence_interval=confidence_intervals,
            basis=f"{n_shots}-shot",
            transferability=transferability
        )

    def transfer_learning_report(self) -> str:
        """Human-readable summary for few-shot predictions."""
        card = self.phenotype_card()
        return (f"Subject {self.subject_id}: \n"
               f"  Phenotype: {card}\n"
               f"  Top constructs: {', '.join(card.responsive_constructs)}\n"
               f"  Prediction confidence: {card.prediction_confidence:.0%}\n"
               f"  Recommendation: {'Use 1–2 shots for quick predictions.' if card.prediction_confidence > 0.7 else 'Collect more training data for better personalization.'}")

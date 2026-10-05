"""The evidence base.

Each consumer-relevant cognitive construct is defined by (a) the canonical brain
regions that index it, (b) the Yeo-7 cortical-network proxies usable with a
surface atlas (Schaefer-1000), (c) a polarity on the approach(+)/avoidance(-)
buy<->sell axis, and (d) the peer-reviewed literature it rests on.

Citations are first-class so every number the detector emits is defensible. The
approach-avoidance composite follows the neuroforecasting framework
(Knutson et al., 2007): reward/value (NAcc, mPFC) drives buying; anterior-insula /
ACC conflict ("pain of paying", regret) drives not-buying.
"""
from __future__ import annotations

from dataclasses import dataclass

YEO7_NETWORKS: tuple[str, ...] = (
    "Visual", "Somatomotor", "DorsalAttention", "VentralAttention",
    "Limbic", "Frontoparietal", "Default",
)


@dataclass(frozen=True)
class Construct:
    key: str
    label: str
    regions: tuple[str, ...]         # canonical anatomical regions (incl. subcortical)
    yeo7_networks: tuple[str, ...]   # cortical-surface proxies (Schaefer-1000 / Yeo-7)
    polarity: float                  # +approach (buy) / -avoidance (sell) weight
    description: str
    citations: tuple[str, ...]
    subcortical: bool = False        # True if key regions are subcortical (cortical proxy used)


CONSTRUCTS: tuple[Construct, ...] = (
    Construct(
        key="reward_value", label="Reward & Value",
        regions=("ventromedial PFC", "orbitofrontal cortex", "ventral striatum / nucleus accumbens"),
        yeo7_networks=("Limbic", "Default"), polarity=0.30, subcortical=True,
        description="Subjective value / approach drive — the core 'buy' signal.",
        citations=(
            "Knutson, Rick, Wimmer, Prelec & Loewenstein (2007). Neural predictors of purchases. Neuron 53:147.",
            "Bartra, McGuire & Kable (2013). The valuation system: a meta-analysis. NeuroImage 76:412.",
            "Cakir, Cakar, Girisken & Yurdakul (2018). Neural correlates of purchase behavior (fNIRS). Eur. J. Marketing 52(1/2):224.",
            "Kapoor, Sahay, Singh, Pammi & Banerjee (2023). Weak brand choices (fMRI). J. Business Research 154:113230.",
        ),
    ),
    Construct(
        key="emotion", label="Emotional Resonance",
        regions=("amygdala", "ventromedial PFC", "limbic cortex"),
        yeo7_networks=("Limbic",), polarity=0.22, subcortical=True,
        description="Affective arousal and somatic-marker valuation.",
        citations=(
            "Vlasceanu (2014). Neuroeconomics and neuromarketing. Procedia SBS 127:758.",
            "Phan, Wager, Taylor & Liberzon (2002). Functional neuroanatomy of emotion: a meta-analysis. NeuroImage 16:331.",
        ),
    ),
    Construct(
        key="attention", label="Attention Capture",
        regions=("intraparietal sulcus", "frontal eye fields", "temporoparietal junction"),
        yeo7_networks=("DorsalAttention",), polarity=0.18,
        description="Orienting and sustained attention toward the stimulus.",
        citations=("Corbetta & Shulman (2002). Control of goal-directed and stimulus-driven attention. Nat Rev Neurosci 3:201.",),
    ),
    Construct(
        key="visual_sensory", label="Visual & Sensory Salience",
        regions=("V1-V4", "ventral & dorsal visual streams"),
        yeo7_networks=("Visual",), polarity=0.10,
        description="Low-level and category-level visual response to the stimulus.",
        citations=(
            "Wandell, Dumoulin & Brewer (2007). Visual field maps in human cortex. Neuron 56:366.",
            "Allen et al. (2022). A massive 7T fMRI dataset (NSD). Nat Neurosci 25:116.",
        ),
    ),
    Construct(
        key="memory_encoding", label="Memory Encoding",
        regions=("hippocampus", "parahippocampal cortex", "default-mode cortex"),
        yeo7_networks=("Default",), polarity=0.10, subcortical=True,
        description="Likelihood the content is encoded and later recalled (brand memory).",
        citations=(
            "Wagner et al. (1998). Building memories: encoding predicts later recall. Science 281:1188.",
            "Raichle (2015). The brain's default mode network. Annu Rev Neurosci 38:433.",
        ),
    ),
    Construct(
        key="conflict_risk", label="Decision Conflict & Risk",
        regions=("anterior cingulate cortex", "anterior insula"),
        yeo7_networks=("VentralAttention",), polarity=-0.22,
        description="Conflict, risk, regret and 'pain of paying' — drives SELL / avoid.",
        citations=(
            "Knutson et al. (2007). Neural predictors of purchases (anterior insula). Neuron 53:147.",
            "Botvinick, Braver, Barch, Carter & Cohen (2001). Conflict monitoring and cognitive control. Psych Rev 108:624.",
            "Kapoor et al. (2023). Weak brand choices: rostral/dorsal ACC (fMRI). J. Business Research 154:113230.",
            "Shang, Deng & Liu (2018). Online purchase intention ERP (N2/N400). NeuroQuantology 16(5):246.",
        ),
    ),
    Construct(
        key="cognitive_load", label="Cognitive Load / Deliberation",
        regions=("dorsolateral PFC", "dorsomedial PFC"),
        yeo7_networks=("Frontoparietal",), polarity=-0.15,
        description="Top-down deliberation / mental effort; high load is decision friction.",
        citations=(
            "Kapoor et al. (2023). Right DLPFC in weak brand choice (fMRI). J. Business Research 154:113230.",
            "Vlasceanu (2014). Dual-process (System-2 deliberation). Procedia SBS 127:758.",
        ),
    ),
)

CONSTRUCTS_BY_KEY = {c.key: c for c in CONSTRUCTS}

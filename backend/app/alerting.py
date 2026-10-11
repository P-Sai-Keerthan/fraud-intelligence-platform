"""
Alert levels: how a model score becomes "Low / Medium / High / Critical Risk".

Two policies exist (decision record: docs/R-02-alert-score-decision.md):

* ``validated_cutoffs`` -- the default random-forest model set. The score is cut at the two operating
  points that were fixed on the validation period of the training data (4C-3B rule) and confirmed,
  unchanged, on fresh hold-outs: the *alert* cut-off (false-positive rate within 1 %) and the *critical*
  cut-off (within 0.1 %). Levels: Low below the alert cut-off, Medium from it, Critical from the second.
  "High Risk" is not emitted: there is no validated third operating point.
* ``legacy_bands`` -- fixed 25/50/80 bands designed for the earlier DNN's output; still used by every other
  model set, which have no approved cut-offs of their own.

Scores here are on the classifier's 0-1 output scale (the API's ``fraud_probability`` is this x 100).
They are model scores, not calibrated probabilities; nothing in this module calibrates them.
"""

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

LEVEL_LOW, LEVEL_MEDIUM, LEVEL_HIGH, LEVEL_CRITICAL = "Low Risk", "Medium Risk", "High Risk", "Critical Risk"
ALL_LEVELS = (LEVEL_LOW, LEVEL_MEDIUM, LEVEL_HIGH, LEVEL_CRITICAL)

VALIDATED = "validated_cutoffs"
LEGACY = "legacy_bands"


@dataclass(frozen=True)
class AlertPolicy:
    kind: str                                   # VALIDATED or LEGACY
    # ((level, inclusive lower bound on the 0-1 score), ...) in increasing order; the first bound is 0
    tiers: Tuple[Tuple[str, float], ...]
    status: str

    def level(self, score: float) -> str:
        chosen = self.tiers[0][0]
        for name, lower in self.tiers:
            if score >= lower:
                chosen = name
        return chosen

    def bands(self) -> dict:
        """Level -> rule text on the 0-100 Fraud Score (the wording /model-info shows). Unused levels are listed
        with an explicit "not used" rule so clients can still render all four."""
        out = {name: "not used by this model set (no validated cut-off)" for name in ALL_LEVELS}
        for i, (name, lower) in enumerate(self.tiers):
            upper = self.tiers[i + 1][1] if i + 1 < len(self.tiers) else None
            lo, hi = _fmt(lower * 100), _fmt(upper * 100) if upper is not None else None
            if i == 0:
                out[name] = f"fraud_probability < {hi}"
            elif hi is None:
                out[name] = f"fraud_probability >= {lo}"
            else:
                out[name] = f"{lo} <= fraud_probability < {hi}"
        return out


def _fmt(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


LEGACY_POLICY = AlertPolicy(
    LEGACY,
    ((LEVEL_LOW, 0.0), (LEVEL_MEDIUM, 0.25), (LEVEL_HIGH, 0.50), (LEVEL_CRITICAL, 0.80)),
    "legacy fixed bands, applied to this model set; not derived from any evaluation of it; "
    "candidate-specific thresholds have not been approved",
)


def validated_policy(frozen_cutoffs: Mapping[str, float]) -> AlertPolicy:
    """Low / Medium / Critical from the frozen validation-period operating points (0-1 scale)."""
    return AlertPolicy(
        VALIDATED,
        ((LEVEL_LOW, 0.0), (LEVEL_MEDIUM, float(frozen_cutoffs["policy_b"])), (LEVEL_CRITICAL, float(frozen_cutoffs["critical"]))),
        "alert cut-offs fixed on the validation period of the training data and confirmed unchanged on fresh "
        "synthetic hold-outs; they are score cut-offs, not probabilities",
    )


def policy_for(model_set_name: str, frozen_cutoffs: Optional[Mapping[str, float]]) -> AlertPolicy:
    """Validated cut-offs for a model set that has them (the default random forest), else the legacy bands."""
    from .model_sets import RF_MODEL_SET
    if model_set_name == RF_MODEL_SET and frozen_cutoffs is not None:
        return validated_policy(frozen_cutoffs)
    return LEGACY_POLICY

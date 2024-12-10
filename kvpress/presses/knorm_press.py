from dataclasses import dataclass, field

from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.scorers.knorm_scorer import KnormScorer


@dataclass
class KnormPress(ScorerPress):
    scorer: KnormScorer = field(default_factory=KnormScorer, init=False)
    compression_ratio: float = 0.0

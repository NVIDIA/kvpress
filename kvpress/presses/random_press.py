from dataclasses import dataclass, field

from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.scorers.random_scorer import RandomScorer


@dataclass
class RandomPress(ScorerPress):
    scorer: RandomScorer = field(default_factory=RandomScorer)
    compression_ratio: float = 0.0

    def __post_init__(self):
        self.scorer = RandomScorer()
        super().__post_init__()

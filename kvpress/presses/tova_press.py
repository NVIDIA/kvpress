import dataclasses

from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.scorers.tova_scorer import TOVAScorer


@dataclasses.dataclass
class TOVAPress(ScorerPress):
    scorer: TOVAScorer = dataclasses.field(default_factory=TOVAScorer, init=False)
    compression_ratio: float = 0.0
    window_size: int = 1

    def __post_init__(self):
        self.scorer = TOVAScorer(window_size=self.window_size)
        super().__post_init__()

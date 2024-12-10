from dataclasses import dataclass, field

from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.scorers.expected_attention_scorer import ExpectedAttentionScorer


@dataclass
class ExpectedAttentionPress(ScorerPress):
    scorer: ExpectedAttentionScorer = field(default_factory=ExpectedAttentionScorer, init=False)
    compression_ratio: float = 0.0
    n_future_positions: int = 512
    n_sink: int = 4
    use_covariance: bool = True
    use_vnorm: bool = True

    def __post_init__(self):
        self.scorer = ExpectedAttentionScorer(
            n_future_positions=self.n_future_positions,
            n_sink=self.n_sink,
            use_covariance=self.use_covariance,
            use_vnorm=self.use_vnorm,
        )
        super().__post_init__()

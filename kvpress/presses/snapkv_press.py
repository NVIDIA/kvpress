from dataclasses import dataclass, field

from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.scorers.snapkv_scorer import SnapKVScorer


@dataclass
class SnapKVPress(ScorerPress):
    scorer: SnapKVScorer = field(default_factory=SnapKVScorer, init=False)
    compression_ratio: float = 0.0
    window_size: int = 64
    kernel_size: int = 5

    def __post_init__(self):
        self.scorer = SnapKVScorer(window_size=self.window_size, kernel_size=self.kernel_size)
        super().__post_init__()

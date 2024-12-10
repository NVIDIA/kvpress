import dataclasses

from kvpress import DefaultPress, TOVAScorer


@dataclasses.dataclass
class TOVAPress(DefaultPress):
    scorer: TOVAScorer = dataclasses.field(default_factory=TOVAScorer)
    compression_ratio: float = 0.0
    window_size: int = 1

    def __post_init__(self):
        self.scorer = TOVAScorer(window_size=self.window_size)
        super().__post_init__()

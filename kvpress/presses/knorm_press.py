from dataclasses import dataclass, field

from kvpress import DefaultPress, KnormScorer


@dataclass
class KnormPress(DefaultPress):
    scorer: KnormScorer = field(default_factory=KnormScorer)
    compression_ratio: float = 0.0

    def __post_init__(self):
        self.scorer = KnormScorer()
        super().__post_init__()

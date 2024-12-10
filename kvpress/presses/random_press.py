from dataclasses import dataclass, field

from kvpress import DefaultPress, RandomScorer


@dataclass
class RandomPress(DefaultPress):
    scorer: RandomScorer = field(default_factory=RandomScorer)
    compression_ratio: float = 0.0

    def __post_init__(self):
        self.scorer = RandomScorer()
        super().__post_init__()

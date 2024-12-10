from dataclasses import dataclass, field

from kvpress import DefaultPress, SnapKVScorer


@dataclass
class SnapKVPress(DefaultPress):
    scorer: SnapKVScorer = field(default_factory=SnapKVScorer)
    compression_ratio: float = 0.0
    window_size: int = 64
    kernel_size: int = 5

    def __post_init__(self):
        self.scorer = SnapKVScorer(
            window_size=self.window_size,
            kernel_size=self.kernel_size,
        )
        super().__post_init__()

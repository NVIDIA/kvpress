from dataclasses import dataclass, field

from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.scorers.streaming_llm_scorer import StreamingLLMScorer


@dataclass
class StreamingLLMPress(ScorerPress):
    scorer: StreamingLLMScorer = field(default_factory=StreamingLLMScorer)
    compression_ratio: float = 0.0
    n_sink: int = 4

    def __post_init__(self):
        self.scorer = StreamingLLMScorer(n_sink=self.n_sink)
        super().__post_init__()

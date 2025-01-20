from dataclasses import dataclass

from kvpress.presses.adakv_press import AdaKVPress
from kvpress.presses.base_press import BasePress
from kvpress.presses.observed_attention_press import ObservedAttentionPress


@dataclass
class ComposedPress(BasePress):
    """
    Chain multiple presses together to create a composed press
    """

    presses: list[BasePress]

    def __post_init__(self):
        self.compression_ratio = None
        assert not any(
            isinstance(press, (ObservedAttentionPress, AdaKVPress)) for press in self.presses
        ), "ComposedPress cannot contains ObservedAttentionPress or AdaKVPress"

    def forward_hook(self, module, input, kwargs, output):
        self.compression_ratio = 1.0
        for press in self.presses:
            output = press.forward_hook(module, input, kwargs, output)
            self.compression_ratio *= press.compression_ratio  # type: ignore
        return output

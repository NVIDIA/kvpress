from dataclasses import dataclass
from kvpress.presses.base_press import BasePress


@dataclass
class ComposedPress(BasePress):
    """
    Chain multiple presses together to create a composed press
    """

    presses: list[BasePress]

    def forward_hook(self, module, input, kwargs, output):
        self.compression_ratio = 1
        for press in self.presses:
            output = press.forward_hook(module, input, kwargs, output)
            self.compression_ratio *= press.compression_ratio
        return output
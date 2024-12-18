# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
from attr import dataclass
from torch import nn

from kvpress import ExpectedAttentionPress
from kvpress.presses.chunk_press import ChunkPress
from tests.fixtures import kv_press_unit_test_pipeline  # noqa: F401


@dataclass
class RecordExpectedAttentionPress(ExpectedAttentionPress):
    compression_ratio: float = 0.0
    n_future_positions: int = 10
    n_sink: int = 0
    use_covariance: bool = True
    use_vnorm: bool = True
    num_called: int = 0

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        self.num_called += 1
        return super().score(module, hidden_states, keys, values, attentions, kwargs)


@torch.no_grad()
def test_chunk_press(kv_press_unit_test_pipeline, caplog):  # noqa: F811
    press = ChunkPress(RecordExpectedAttentionPress(compression_ratio=0.4))
    context = "This is a test context." * 20
    question = "What is the answer to life, the universe, and everything?"
    kv_press_unit_test_pipeline(context, question=question, press=press, chunk_size=10, max_new_token=10)["answer"]
    assert press.press.num_called > 5

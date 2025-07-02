# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class LagKVPress(ScorerPress):
    """
    LagKV: Lag-relative information-based KV cache compression.
    
    Based on LagKV (https://arxiv.org/abs/2504.04704), this method compresses
    the KV cache by leveraging lag-relative information between different
    partitions of the sequence. The approach divides the sequence into
    partitions and uses subsequent partitions as references for scoring
    tokens in prior partitions.
    
    The method works by:
    1. Preserving the first n_sink tokens (attention sinks)
    2. Dividing the remaining sequence into partitions of size lag_size
    3. Using each partition as a reference to score tokens in the previous partition
    4. Computing importance scores based on the lag-relative information
    5. Optionally enabling cross-partition scoring for more flexible allocation
    
    This approach is particularly effective because:
    - It captures temporal dependencies between different parts of the sequence
    - It provides a structured way to identify important tokens across partitions
    - It can adapt to different attention patterns within each partition
    - It maintains computational efficiency through partitioned processing
    
    The lag-relative scoring helps identify tokens that are important for
    maintaining coherence across different temporal segments of the input.
    """

    compression_ratio: float = 0.0
    """
    Fraction of key-value pairs to remove during compression.
    See ScorerPress.compression_ratio for detailed description.
    """
    
    n_sink: int = 4
    """
    Number of initial tokens to preserve as attention sinks.
    
    These tokens at the beginning of the sequence are never pruned and serve
    as attention sinks that help maintain model stability. The sink tokens
    are excluded from the partitioning and lag-relative scoring process.
    
    See StreamingLLMPress.n_sink for detailed description of sink tokens.
    """
    
    lag_size: int = 128
    """
    Size of each partition for lag-relative scoring.
    
    The sequence (after sink tokens) is divided into non-overlapping partitions
    of this size. Each partition serves as a reference for scoring tokens in
    the previous partition, creating a lag-relative information flow.
    
    Larger partition sizes:
    - Provide more context for lag-relative scoring
    - May capture longer-range dependencies within partitions
    - Reduce the number of partitions to process
    
    Smaller partition sizes:
    - Create more fine-grained lag-relative relationships
    - May be more responsive to local attention patterns
    - Increase the number of partition boundaries
    
    The default of 128 provides a good balance between context and granularity.
    """
    
    cross_scoring: bool = False
    """
    Whether to enable cross-partition scoring (experimental feature).
    
    When True, the scoring is not limited to within-partition relationships
    and can consider cross-partition dependencies. This allows for more
    flexible token allocation across the entire sequence.
    
    - False: Limit scoring to within-partition relationships (default)
    - True: Enable cross-partition scoring for more flexible allocation
    
    Cross-scoring can be particularly useful when combined with wrapper
    methods like AdaKVPress that need to allocate tokens across attention heads.
    
    Note: This is an experimental feature and may affect compression behavior.
    """

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        bsz, num_key_value_heads, q_len, d = keys.shape
        if q_len < self.n_sink + 2 * self.lag_size:
            # no compression
            score = torch.ones((bsz, num_key_value_heads, q_len),
                               dtype=keys.dtype, device=keys.device)
            if q_len > self.n_sink:
                # make sure the sliding part will be selected.
                score[:, :, self.n_sink:] = (torch.arange(q_len - self.n_sink, device=keys.device)
                                             / (q_len - self.n_sink)
                                             ).to(keys.dtype)
            return score

        end_idx = self.n_sink + ((q_len - self.n_sink) // self.lag_size) * self.lag_size
        tail_len = self.lag_size + q_len - end_idx

        key_score = self._get_states_score(
            keys[:, :, self.n_sink:end_idx].view(bsz, num_key_value_heads, -1, self.lag_size, d))
        value_score = self._get_states_score(
            values[:, :, self.n_sink:end_idx].view(bsz, num_key_value_heads, -1, self.lag_size, d))
        # score is in range [0, 1]
        score = (key_score + value_score) / 2

        if not self.cross_scoring:
            score = score.argsort(dim=-1).argsort(dim=-1) / self.lag_size
            score = score.to(keys.dtype)
        # the parts should always keep
        sink_shape = (bsz, num_key_value_heads, self.n_sink)
        sink_score = torch.ones(sink_shape, dtype=score.dtype, device=score.device)
        tail_shape = (bsz, num_key_value_heads, tail_len)
        tail_score = torch.ones(tail_shape, dtype=score.dtype, device=score.device)
        score = torch.cat((sink_score, score.reshape(bsz, num_key_value_heads, -1), tail_score), dim=-1)
        return score

    def _get_states_score(self, target_v):
        """evaluate the scores of keys and values for each token"""
        ref = target_v[:, :, 1:, :, :]
        v = target_v[:, :, :-1, :, :]
        # lag-relative information
        min_r = ref.min(dim=-2).values.unsqueeze(-2).expand(-1, -1, -1, self.lag_size, -1)
        max_r = ref.max(dim=-2).values.unsqueeze(-2).expand(-1, -1, -1, self.lag_size, -1)

        score = ((v - min_r) / (max_r - min_r)).std(dim=-1).softmax(dim=-1)
        return score

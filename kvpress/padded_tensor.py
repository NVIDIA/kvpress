# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

FILL_VALUE = 0.0


class PaddedTensor:
    """A 4D tensor (batch, heads, seq, head_dim) with ragged seq dimension.

    Valid data is packed at the front of dim 2 (positions 0..length-1 per head).
    Padding positions may contain stale data; call fill_padding() or to_dense()
    to materialise FILL_VALUE there.

    Designed for KV cache storage in the filtering press.
    """

    def __init__(self, data: torch.Tensor, lengths: torch.Tensor):
        assert data.ndim == 4  # (batch, heads, seq, head_dim)
        assert lengths.shape == data.shape[:2]  # (batch, heads)
        self.data = data
        self.lengths = lengths

    @property
    def max_length(self) -> int:
        return self.data.shape[2]

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    def valid_mask(self, include_last=False) -> torch.Tensor:
        """Boolean mask of valid (non-padding) positions: shape (batch, heads, seq)."""
        indices = torch.arange(self.max_length, device=self.device)
        mask = indices < self.lengths.unsqueeze(-1)
        if include_last and self.max_length > 0:
            mask[:, :, -1] = True
        return mask

    def fill_padding(self, fill_value: float = FILL_VALUE) -> None:
        """Fill positions beyond valid lengths in-place."""
        mask = ~self.valid_mask().unsqueeze(-1).expand_as(self.data)
        self.data[mask] = fill_value

    def accept_last(self, accepted: torch.Tensor) -> None:
        """Incorporate the last position into the valid prefix for accepted heads.

        For heads where there is a gap between the valid prefix and the last position
        (lengths < max_length - 1), copies data from the last position to
        position lengths[head] (first slot after the valid prefix).
        Increments lengths for all accepted heads where lengths < max_length.

        accepted: boolean (batch, heads)
        """
        last_pos = self.max_length - 1
        needs_swap = accepted & (self.lengths < last_pos)
        if needs_swap.any():
            b_idx, h_idx = needs_swap.nonzero(as_tuple=True)
            target_pos = self.lengths[b_idx, h_idx]
            self.data[b_idx, h_idx, target_pos] = self.data[b_idx, h_idx, last_pos]

        needs_increment = accepted & (self.lengths <= last_pos)
        self.lengths = self.lengths + needs_increment.long()

    def remove_last(self, head_bitset: torch.Tensor) -> None:
        """Decrement lengths for selected heads.

        head_bitset: boolean (batch, heads)
        """
        self.lengths = (self.lengths - head_bitset.long()).clamp_(min=0)

    def shrink(self) -> None:
        """Trim backing tensor to the maximum valid length across all heads."""
        max_valid = int(self.lengths.max().item()) if self.lengths.numel() > 0 else 0
        if max_valid < self.max_length:
            self.data = self.data[:, :, :max_valid, :].contiguous()

    def clone(self) -> PaddedTensor:
        """Create an independent deep copy."""
        return PaddedTensor(self.data.clone(), self.lengths.clone())

    def __repr__(self) -> str:
        return f"PaddedTensor(shape={list(self.data.shape)}, lengths={self.lengths})"

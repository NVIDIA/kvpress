# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from kvpress.presses.base_press import BasePress
from kvpress.presses.snapkv_press import SnapKVPress
from kvpress.presses.key_rerotation_press import KeyRerotationPress


@dataclass
class FinchPress(BasePress):
    """
    FINCH: Prompt-guided Key-Value Cache Compression.

    SnapKV-style compression with dynamic window sizing based on delimiter tokens.
    Requires input format: `context + delimiter_token + question`. The delimiter
    separates context from query, allowing dynamic window size determination.

    Use `update_model_and_tokenizer` method to set delimiter token before use.


    Based on FINCH (https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00716/125280).

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    chunk_length : int, optional
        Length of chunks for optional chunked compression. None processes entire context at once.
    normalize_scores : bool, default=True
        Whether to normalize attention scores by number of non-zero weights.
    rerotate_keys : bool, default=True
        Whether to rerotate keys after compression using RoPE for proper positional encoding.
    delimiter_token : str
        Delimiter token string separating context from query (set automatically).
    delimiter_token_id : int
        Token ID for delimiter token (set automatically).
    window_size : int
        Dynamically determined window size based on delimiter position (set automatically).
    """

    compression_ratio: float = 0.0
    chunk_length: int = None
    normalize_scores: bool = True
    rerotate_keys: bool = True
    delimiter_token: str = field(default=None, init=False)
    delimiter_token_id: int = field(default=None, init=False)
    window_size: int = field(default=None, init=False)

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        """
        Similar to SnapKVPress except it adds a normalization step before averaging on the context window.
        """

        bsz, num_key_value_heads, q_len, _ = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        if attentions is not None:
            attn_weights = attentions[..., -self.window_size :, : -self.window_size]
        else:
            attn_weights = SnapKVPress.compute_window_attention(
                module, hidden_states, keys, self.window_size, kwargs["position_embeddings"]
            )

        if self.normalize_scores:
            non_zero_counts = torch.arange(q_len - self.window_size, q_len)[None, None, :, None]
            non_zero_counts = non_zero_counts.to(attn_weights.device)
            attn_weights = attn_weights * non_zero_counts

        # Average per group
        scores = attn_weights.mean(dim=-2)
        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, q_len - self.window_size)
        scores = scores.mean(dim=2)

        # Add back the observation window. Use max score to make sure the window is not pruned.
        scores = F.pad(scores, (0, self.window_size), value=scores.max().item())
        return scores

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        """
        Scores are computed by chunks, keys and values are then compressed and re-rotated.
        """

        if self.compression_ratio == 0:
            return keys, values
        assert self.window_size is not None, "window_size must be provided"

        # Compute scores
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

        # Compute indices to keep (optionally by chunks)
        q_len = hidden_states.shape[1]
        if self.chunk_length is None:
            n_kept = int(q_len * (1 - self.compression_ratio))
            indices = scores.topk(n_kept, dim=-1).indices
        else:
            assert self.chunk_length > self.window_size / (1 - self.compression_ratio)
            indices = []
            for i in range(0, q_len, self.chunk_length):
                chunk_scores = scores[:, :, i : i + self.chunk_length]
                n_kept = max(1, int(chunk_scores.shape[2] * (1 - self.compression_ratio)))
                chunk_indices = i + chunk_scores.topk(n_kept, dim=-1).indices
                indices.append(chunk_indices)
            indices = torch.cat(indices, dim=-1)
        if self.rerotate_keys:
            indices = torch.sort(indices, dim=2).values
            keys = KeyRerotationPress.rerotate_keys(module, indices, keys)
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        else:
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
            keys = keys.gather(2, indices).contiguous()

        values = values.gather(2, indices).contiguous()

        return keys, values

    def embed_token_forward_hook(self, module, input, output):
        """
        Forward hook to detect a delimiter token between the context and the window
        """
        if input[0].shape[1] > 1 and self.delimiter_token_id in input[0][0]:  # prefilling
            assert len(input[0]) == 1, "Only batch size 1 is supported."
            # Find the delimiter token and compute the window size
            delim_tokens = input[0][0] == self.delimiter_token_id
            assert delim_tokens.sum() == 1, "Only one delimiter token should be present."
            context_length = int(torch.nonzero(delim_tokens)[0].item())
            self.window_size = len(input[0][0]) - 1 - context_length
            assert self.window_size > 0, "No window detected (window size must be > 0)."
            # Remove the delimiter token from the output
            output = output[:, ~delim_tokens]
        return output

    def update_model_and_tokenizer(self, model, tokenizer, delimiter_token: str = "<|finch_sep|>"):
        """
        Set the delimiter token and update the tokenizer accordingly.
        This method should be called before calling the press.
        """
        self.delimiter_token = delimiter_token
        if delimiter_token not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [delimiter_token]})
        self.delimiter_token_id = tokenizer.convert_tokens_to_ids(delimiter_token)  # type: ignore
        # update model embeddings
        model.resize_token_embeddings(len(tokenizer))
        return tokenizer

    @contextmanager
    def __call__(self, model):
        # The user should set the delimiter_token_id before calling the press.
        if self.delimiter_token_id is None:
            raise ValueError(
                """No delimiter token ID provided.
                             Use the update_model_and_tokenizer method before calling the press."""
            )

        with super().__call__(model):
            try:
                hook = model.model.embed_tokens.register_forward_hook(self.embed_token_forward_hook)
                yield
            finally:
                hook.remove()

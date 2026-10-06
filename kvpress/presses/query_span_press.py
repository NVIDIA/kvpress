# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import rotate_half

from kvpress.presses.base_press import BasePress, is_prefilling
from kvpress.utils import compute_n_kept, extract_keys_and_values, get_prerope_query_states

logger = logging.getLogger(__name__)


def forward_propagate(scores: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Decaying running max along the sequence: out[t] = max_{j <= t} gamma^(t - j) * scores[j].
    Computed in closed form as a cumulative max in log space.
    """
    seq_len = scores.shape[-1]
    log_gamma = math.log(gamma)
    positions = torch.arange(seq_len, device=scores.device, dtype=torch.float32)
    log_scores = scores.float().clamp_min(1e-30).log()
    return (torch.cummax(log_scores - positions * log_gamma, dim=-1).values + positions * log_gamma).exp()


@dataclass
class QuerySpanPress(BasePress):
    """
    QuerySpan: query-aware, single-pass KV cache compression with surprisal-gated span propagation.

    Designed for query-aware compression, i.e. the question is the last part of the compressed context
    (`query_aware=True` in the evaluation). Importance is computed in three steps, all during the single prefill:

    1. Question attention: for every layer and KV head, the attention of the last `window_size` queries
       (the question) over the context, aggregated with a max over queries and over the query heads of the
       group, and normalized as in KVzip+ (divided by ||h|| of the query, multiplied by ||W_o v||).
    2. Span propagation: the question typically attends to the beginning of the relevant span, but decoding
       copies the whole span token by token (induction), so every continuation token is needed too.
       Importance is carried forward with a decay `gamma`: c(t) = max(s(t), gamma * c(t-1)).
    3. Surprisal gating: carried importance is only granted to tokens the model cannot predict,
       s(t) = max(s(t), c(t) * u(t) ** alpha), with u(t) = 1 - p(x_t | x_<t) read from the prefill's own
       next-token distribution. u is max-pooled over `u_pool` tokens so that predictable tokens inside an
       unpredictable span (e.g. separators inside an identifier) are kept.

    The budget is then allocated globally across layers and heads (as in KVzip), and pruned pairs are masked
    through `module.masked_key_indices` (see attention_patch.py). The first `n_sink` tokens are always kept.
    Unlike KVzip, no additional forward pass of the model is required: the only extra compute is the
    question attention and one lm_head projection of the context.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    window_size : int, default=64
        Number of last tokens of the context used as queries (should cover the question).
    n_sink : int, default=4
        Number of initial tokens always kept.
    gamma : float, default=0.95
        Decay of the forward span propagation.
    alpha : float, default=0.5
        Exponent applied to the unpredictability u(t) when gating propagated importance.
    u_pool : int, default=3
        Kernel size of the max-pooling applied to u(t). Must be odd.
    lm_head_chunk_size : int, default=1024
        Number of tokens projected at once through the lm_head to compute u(t).
    """

    compression_ratio: float = 0.0
    window_size: int = 64
    n_sink: int = 4
    gamma: float = 0.95
    alpha: float = 0.5
    u_pool: int = 3
    lm_head_chunk_size: int = 1024

    scores: dict = field(init=False, default_factory=dict, repr=False)
    modules: dict = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"
        assert 0 < self.gamma < 1, "gamma must be in (0, 1)"
        assert self.u_pool % 2 == 1, "u_pool must be odd"

    @staticmethod
    def value_output_norm(module: nn.Module, values: torch.Tensor) -> torch.Tensor:
        """
        ||W_o v|| for each query head of the group, computed via the Gram matrix of W_o.
        Shape (bsz, num_kv_heads, num_kv_groups, seq_len).
        """
        num_kv_heads, head_dim = values.shape[1], values.shape[-1]
        num_groups = module.config.num_attention_heads // num_kv_heads
        Wo = module.o_proj.weight.view(module.config.hidden_size, num_kv_heads, num_groups, head_dim).float()
        gram = torch.einsum("jhgi,jhgk->hgik", Wo, Wo)
        v = values.float()
        return torch.einsum("bhti,hgik,bhtk->bhgt", v, gram, v).clamp_min(0).sqrt()

    def question_scores(
        self, module: nn.Module, hidden_states: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, kwargs: dict
    ) -> torch.Tensor:
        """
        Normalized attention of the last `window_size` queries, max over queries and group heads.
        Shape (bsz, num_kv_heads, seq_len).
        """
        bsz, num_kv_heads, k_len, head_dim = keys.shape
        num_groups = module.config.num_attention_heads // num_kv_heads
        window_size = min(self.window_size, k_len)

        queries = get_prerope_query_states(module, hidden_states[:, -window_size:])
        cos, sin = kwargs["position_embeddings"]
        cos, sin = cos[:, -window_size:].unsqueeze(1), sin[:, -window_size:].unsqueeze(1)
        queries = (queries * cos) + (rotate_half(queries) * sin)
        queries = queries.view(bsz, num_kv_heads, num_groups, window_size, head_dim)

        attn = torch.einsum("bhgqd,bhkd->bhgqk", queries.float(), keys.float()) / math.sqrt(head_dim)
        causal_mask = torch.ones(window_size, k_len, dtype=torch.bool, device=keys.device)
        attn.masked_fill_(causal_mask.triu(k_len - window_size + 1), float("-inf"))
        attn = attn.softmax(dim=-1)

        h_norm = hidden_states[:, -window_size:].float().norm(dim=-1)
        attn = attn / h_norm[:, None, None, :, None]
        attn = attn * self.value_output_norm(module, values)[:, :, :, None, :]
        return attn.amax(dim=(2, 3))

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """Only compute and store scores; compression is applied once all layers have been scored."""
        hidden_states = kwargs["hidden_states"]
        if self.compression_ratio == 0 or not is_prefilling(kwargs["cache_position"], hidden_states.shape[1]):
            return output
        keys, values = extract_keys_and_values(kwargs["past_key_values"], module.layer_idx)
        self.scores[module.layer_idx] = self.question_scores(module, hidden_states, keys, values, kwargs)
        self.modules[module.layer_idx] = module
        return output

    @torch.no_grad()
    def unpredictability(self, model: PreTrainedModel, hidden_states: torch.Tensor, input_ids: torch.Tensor):
        """u(t) = 1 - p(x_t | x_<t) from the final hidden states of the prefill. Shape (bsz, seq_len)."""
        bsz, seq_len, _ = hidden_states.shape
        u = torch.ones(bsz, seq_len, device=hidden_states.device)
        for start in range(0, seq_len - 1, self.lm_head_chunk_size):
            end = min(start + self.lm_head_chunk_size, seq_len - 1)
            log_probs = model.lm_head(hidden_states[:, start:end]).float().log_softmax(dim=-1)
            u[:, start + 1 : end + 1] = 1 - log_probs.gather(-1, input_ids[:, start + 1 : end + 1, None])[..., 0].exp()
        if self.u_pool > 1:
            u = F.max_pool1d(u[:, None], self.u_pool, stride=1, padding=self.u_pool // 2)[:, 0]
        return u

    def compress_all(self, u: torch.Tensor):
        """Propagate, gate and allocate the budget globally across layers and heads."""
        layer_indices = sorted(self.scores)
        gate = u.pow(self.alpha)[:, None, :]
        scores = []
        for layer_idx in layer_indices:
            s = self.scores[layer_idx].float()
            scores.append(torch.maximum(s, forward_propagate(s, self.gamma) * gate))
        scores = torch.stack(scores, dim=1)  # (bsz, n_layers, num_kv_heads, seq_len)
        scores[..., : self.n_sink] = float("inf")

        bsz, n_layers, num_kv_heads, seq_len = scores.shape
        n_pruned = n_layers * num_kv_heads * (seq_len - compute_n_kept(seq_len, self.compression_ratio))
        pruned = torch.topk(-scores.reshape(bsz, -1), n_pruned, dim=-1).indices  # (bsz, n_pruned)
        batch_indices = torch.arange(bsz, device=pruned.device)[:, None].expand_as(pruned)
        layer_of_pruned = pruned // (num_kv_heads * seq_len)
        for i, layer_idx in enumerate(layer_indices):
            mask = layer_of_pruned == i
            flat = pruned[mask] % (num_kv_heads * seq_len)
            self.modules[layer_idx].masked_key_indices = (batch_indices[mask], flat // seq_len, flat % seq_len)

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        self.scores, self.modules = {}, {}
        captured = {}
        language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model

        def capture_input_ids(module, args, kwargs):
            if kwargs.get("input_ids") is not None:
                captured["input_ids"] = kwargs["input_ids"]

        def capture_final_hidden_states(module, input, output):
            captured["hidden_states"] = output

        hooks = [
            language_model.register_forward_pre_hook(capture_input_ids, with_kwargs=True),
            language_model.norm.register_forward_hook(capture_final_hidden_states),
        ]
        try:
            with super().__call__(model):
                yield
            if self.scores:
                if "input_ids" not in captured:
                    raise ValueError("QuerySpanPress requires input_ids (inputs_embeds are not supported)")
                u = self.unpredictability(model, captured["hidden_states"], captured["input_ids"])
                self.compress_all(u)
        finally:
            for hook in hooks:
                hook.remove()
            self.scores, self.modules = {}, {}

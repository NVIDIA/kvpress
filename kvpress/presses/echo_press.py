# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from types import MethodType
from typing import Generator, List

import torch
from transformers import AutoTokenizer, Gemma3PreTrainedModel, PreTrainedModel, PreTrainedTokenizer
from transformers.models.llama.modeling_llama import rotate_half
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from kvpress.presses.base_press import SUPPORTED_MODELS
from kvpress.presses.echo_press_kernels import HAS_TRITON, virtual_scores_triton
from kvpress.presses.kvzip_press import KVzipPress
from kvpress.utils import extract_keys_and_values

logger = logging.getLogger(__name__)

QUERY_BLOCK = 1024  # Limit the materialized attention block size.


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [b, h, t, d]; cos/sin: [b, t, d]"""
    return x * cos.unsqueeze(1) + rotate_half(x) * sin.unsqueeze(1)


def _scores_from_logits(s_sink, s_cached, s_copy, causal):
    """Return each cached key's maximum attention over the query block."""
    s_cached = s_cached.float()
    s_copy = s_copy.float().masked_fill(causal, float("-inf"))
    row_max = torch.maximum(s_cached.amax(-1, keepdim=True), s_copy.amax(-1, keepdim=True))
    if s_sink is not None:
        s_sink = s_sink.float()
        row_max = torch.maximum(row_max, s_sink.amax(-1, keepdim=True))
    p_cached = (s_cached - row_max).exp()
    denom = p_cached.sum(-1, keepdim=True) + (s_copy - row_max).exp().sum(-1, keepdim=True)
    if s_sink is not None:
        denom = denom + (s_sink - row_max).exp().sum(-1, keepdim=True)
    return (p_cached / denom).amax(dim=(0, 2, 3))


_SCORERS: dict = {}


def _get_scorer(compile_scoring: bool):
    """Compile the scoring step once, falling back to eager mode."""
    if not compile_scoring:
        return _scores_from_logits
    if "compiled" not in _SCORERS:
        try:
            _SCORERS["compiled"] = torch.compile(_scores_from_logits, dynamic=True)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"torch.compile unavailable ({e}); using the eager scoring step")
            _SCORERS["compiled"] = _scores_from_logits
    return _SCORERS["compiled"]


@dataclass
class EchoPress(KVzipPress):
    """
    Approximate KVzip scores during prefill using virtual context repetition.

    RoPE composition lets EchoPress score a repeated chunk from its prefill queries and
    cached keys, avoiding reconstruction passes for later chunks. EchoPress reconstructs
    the first chunk exactly and uses its paired virtual and exact scores to calibrate later
    virtual scores independently for every request, layer, and KV head. Eviction behavior
    is inherited from KVzipPress.

    Only batch size 1 and a single full-context prefill are supported.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    layerwise : bool, default=False
        Whether to enable uniform compression ratios across layers (see KVzipPress).
    n_sink : int, default=4
        Number of initial tokens to preserve as attention sinks.
    chunk_size : int, default=2048
        Total length of the first exact pass and size of later virtual chunks.
    exact_first_chunk : bool, default=True
        Re-score the first chunk with one reconstruction pass. Disable only to reproduce
        the legacy all-virtual variant.
    guard_tokens : int, default=0
        Number of tokens to retain at each chunk boundary.
    score_calibration : bool, default=True
        Apply a per-request score map fitted on the first chunk.
    calibration_direction : str, default="virtual_to_exact"
        Direction of the per-sample quantile map: "exact_to_virtual" maps the
        exact first chunk onto the virtual scale, while "virtual_to_exact" maps
        later virtual chunks onto the exact first-chunk scale.
    calibration_scope : str, default="head"
        Quantile-map granularity: "global", "layer", or "head".
    scoring_backend : str, default="auto"
        One of "auto", "triton", "torch", or "torch_eager".
    """

    chunk_size: int = 2048
    exact_first_chunk: bool = True
    guard_tokens: int = 0
    score_calibration: bool = True
    calibration_direction: str = "virtual_to_exact"
    calibration_scope: str = "head"
    scoring_backend: str = "auto"
    _q: dict = field(default_factory=dict, repr=False)  # Prefill queries by layer.
    _v0: dict = field(default_factory=dict, repr=False)  # First-chunk virtual scores by layer.
    _chunks: list = field(default_factory=list, repr=False)  # (start, end, prompt length)
    _pairs: list = field(default_factory=list, repr=False)  # (chunk IDs, repeated IDs)

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"
        assert self.scoring_backend in ("auto", "triton", "torch", "torch_eager"), "unknown scoring_backend"
        assert self.calibration_direction in (
            "exact_to_virtual",
            "virtual_to_exact",
        ), "unknown calibration_direction"
        assert self.calibration_scope in ("global", "layer", "head"), "unknown calibration_scope"
        self._reset_internal_parameters()

    def _init_prompt_ids(self, tokenizer: PreTrainedTokenizer):
        """Find the chat template's prefix and suffix."""
        if tokenizer.chat_template is None:
            prefix_text, suffix_text = "", "\n"
        else:
            dummy_context = "dummy context"
            separator = "\n" + "#" * len(dummy_context)
            temp_context = tokenizer.apply_chat_template(
                [{"role": "user", "content": dummy_context + separator}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            context, suffix_text = temp_context.split(separator)
            prefix_text = context.split(dummy_context)[0]
        self.prefix_length = tokenizer.encode(prefix_text, return_tensors="pt", add_special_tokens=False).shape[-1]
        self._suffix_ids = tokenizer.encode(suffix_text, return_tensors="pt", add_special_tokens=False)

    def prepare(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prev_postfix_size=8,
    ) -> List[tuple[torch.Tensor, torch.Tensor]]:
        """Build chunks, keeping the first repeated sequence at `chunk_size`."""
        ctx_ids = self._context_ids[:, self.prefix_length :].to("cpu")
        self.score_val = torch.zeros(
            (model.config.num_hidden_layers, 1, model.config.num_key_value_heads, self.context_length),
            dtype=model.dtype,
            device=model.device,
        )
        self.score_val[..., : self.n_sink] = 1.0

        prompts = [
            "\n\nRepeat the previous context exactly.",
            "\n\nRepeat the part of the previous context exactly, starting with",
        ]
        q_ids = [tokenizer.encode(p, return_tensors="pt", add_special_tokens=False) for p in prompts]
        first_size = max(1, self.chunk_size - q_ids[0].shape[1] - self._suffix_ids.shape[1])
        chunks = [ctx_ids[:, :first_size]] + self._chunk_fn(ctx_ids[:, first_size:], self.chunk_size)
        chunks = [c for c in chunks if c.shape[1] > 0]
        pairs = []
        for i, a_ids in enumerate(chunks):
            if i == 0:
                prompt_ids = q_ids[0]
            else:
                prompt_ids = torch.cat([q_ids[1], chunks[i - 1][:, -prev_postfix_size:]], dim=1)
            pairs.append((a_ids, torch.cat([prompt_ids, self._suffix_ids, a_ids], dim=1)))
        return pairs

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        if not isinstance(model, SUPPORTED_MODELS):
            logger.warning(f"Model {type(model)} not tested, supported models: {SUPPORTED_MODELS}")
        if isinstance(model, Gemma3PreTrainedModel):
            raise ValueError("EchoPress is not supported for Gemma3ForCausalLM")
        if not all(hasattr(layer.self_attn, "q_proj") for layer in model.model.layers):
            raise NotImplementedError("EchoPress requires attention modules with a q_proj")

        self.post_init_from_model(model)
        tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
        self._init_prompt_ids(tokenizer)
        self._inv_freq = model.model.rotary_emb.inv_freq

        original_forward = model.model.forward

        def wrapped_forward(model_self, *args, **kwargs):
            # Prepare chunks before the layer hooks run.
            self._context_ids = kwargs["input_ids"]
            self._cache = kwargs["past_key_values"]
            self.context_length = self._context_ids.shape[1]
            self._pairs = self.prepare(model, tokenizer)
            start, self._chunks = self.prefix_length, []
            for chunk_ids, repeat_ids in self._pairs:
                self._chunks.append((start, start + chunk_ids.shape[1], repeat_ids.shape[1] - chunk_ids.shape[1]))
                start += chunk_ids.shape[1]
            return original_forward(*args, **kwargs)

        model.model.forward = MethodType(wrapped_forward, model.model)
        handles = []
        if self.compression_ratio > 0:
            for layer_idx, layer in enumerate(model.model.layers):
                handles.append(layer.self_attn.q_proj.register_forward_hook(partial(self._q_hook, layer_idx=layer_idx)))
                handles.append(layer.self_attn.register_forward_hook(self._prefill_hook, with_kwargs=True))
        try:
            try:
                yield
            finally:
                model.model.forward = original_forward
                for handle in handles:
                    handle.remove()
            if self.compression_ratio > 0 and self._context_ids is not None:
                with torch.no_grad():
                    self._finalize(model)
                self.compress_post(model)
        finally:
            self._q.clear()
            self._v0.clear()
            self._chunks, self._pairs = [], []
            self._reset_internal_parameters()

    def _q_hook(self, module, args, output, layer_idx: int):
        """Save projected queries until the attention hook runs."""
        self._q[layer_idx] = output

    def _prefill_hook(self, module, args, kwargs, output):
        """Compute each chunk's virtual scores for one layer."""
        layer_idx = int(module.layer_idx)
        q = self._q.pop(layer_idx)
        assert q.shape[0] == 1, "EchoPress only supports batch size 1"
        assert q.shape[1] == self.context_length, "EchoPress requires a single prefill forward"
        q = q.view(1, q.shape[1], module.config.num_attention_heads, module.head_dim).transpose(1, 2)
        if isinstance(module, Qwen3Attention):
            q = module.q_norm(q)
        cos, sin = kwargs["position_embeddings"]
        q = _rope(q, cos, sin)  # [1, H, n, d]
        keys, _ = extract_keys_and_values(self._cache, layer_idx)  # [1, Hkv, n, d]
        with torch.no_grad():
            for i, (start, end, n_prompt) in enumerate(self._chunks):
                scores = self._chunk_scores(module, q, keys, start, end, n_prompt)
                self.score_val[layer_idx, 0, :, start:end] = scores.to(self.score_val.dtype)
                if i == 0:
                    self._v0[layer_idx] = scores

    def _rotate(self, x: torch.Tensor, delta: int) -> torch.Tensor:
        """Apply a RoPE rotation for a constant position offset."""
        freqs = delta * self._inv_freq.float()
        emb = torch.cat([freqs, freqs])
        return x * emb.cos() + rotate_half(x) * emb.sin()

    def _chunk_scores(self, module, q: torch.Tensor, keys: torch.Tensor, start: int, end: int, n_prompt: int):
        """Return virtual KVzip scores for the chunk `[start, end)`."""
        m = end - start
        sink = min(self.n_sink, start)
        num_heads_kv = keys.shape[1]
        num_groups = q.shape[1] // num_heads_kv
        head_dim = module.head_dim
        delta = self.context_length + n_prompt - start
        # Match KVzip's bf16 logits; normalization remains fp32.
        q1 = q[:, :, start:end].view(1, num_heads_kv, num_groups, m, head_dim)
        q2 = self._rotate(q1.float(), delta).to(q1.dtype)
        backend = self.scoring_backend
        if backend == "auto":
            backend = "triton" if HAS_TRITON and q.is_cuda and head_dim in (64, 128, 256) else "torch"
        if backend == "triton":
            return virtual_scores_triton(q1[0], q2[0], keys[0, :, start:end], keys[0, :, :sink])
        k_sink = keys[:, :, :sink].unsqueeze(2).transpose(-2, -1)  # [1, Hkv, 1, d, sink]
        k_chunk = keys[:, :, start:end].unsqueeze(2).transpose(-2, -1)  # [1, Hkv, 1, d, m]
        q1 = q1 * (1.0 / math.sqrt(head_dim))
        q2 = q2 * (1.0 / math.sqrt(head_dim))
        scorer = _get_scorer(backend == "torch")
        scores = torch.zeros(num_heads_kv, m, device=q.device)
        cols = torch.arange(m, device=q.device)[None, :]
        for j0 in range(0, m, QUERY_BLOCK):
            j1 = min(j0 + QUERY_BLOCK, m)
            causal = cols > torch.arange(j0, j1, device=q.device)[:, None]
            s_sink = torch.matmul(q2[:, :, :, j0:j1], k_sink) if sink > 0 else None
            s_cached = torch.matmul(q2[:, :, :, j0:j1], k_chunk)
            s_copy = torch.matmul(q1[:, :, :, j0:j1], k_chunk)
            scores = torch.maximum(scores, scorer(s_sink, s_cached, s_copy, causal))
        return scores

    @staticmethod
    def _quantile_map(source: torch.Tensor, target: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Quantile-map each row of `values` from `source` to `target`."""
        vs, ts = source.sort(-1).values, target.sort(-1).values
        m0 = vs.shape[-1]
        x = torch.minimum(torch.maximum(values, vs[:, :1]), vs[:, -1:])
        idx = torch.searchsorted(vs, x).clamp(1, m0 - 1)
        x0, x1 = vs.gather(1, idx - 1), vs.gather(1, idx)
        y0, y1 = ts.gather(1, idx - 1), ts.gather(1, idx)
        w = ((x - x0) / (x1 - x0).clamp_min(1e-12)).clamp(0, 1)
        return y0 + w * (y1 - y0)

    def _apply_per_sample_calibration(self, start: int, end: int):
        """Fit on paired first-chunk scores and apply the requested map direction."""
        exact = self.score_val[:, 0, :, start:end].float()
        virtual = torch.stack([self._v0[layer_idx] for layer_idx in range(exact.shape[0])])
        n_layers, n_kv, _ = exact.shape
        rows = {"global": 1, "layer": n_layers, "head": n_layers * n_kv}[self.calibration_scope]
        source, target = (exact, virtual) if self.calibration_direction == "exact_to_virtual" else (virtual, exact)
        if self.calibration_direction == "exact_to_virtual":
            values = exact
            value_slice = slice(start, end)
        else:
            value_start = self._chunks[1][0]
            values = self.score_val[:, 0, :, value_start:].float()
            value_slice = slice(value_start, None)
        mapped = self._quantile_map(source.reshape(rows, -1), target.reshape(rows, -1), values.reshape(rows, -1))
        self.score_val[:, 0, :, value_slice] = mapped.view_as(values).to(self.score_val.dtype)

    def _finalize(self, model: PreTrainedModel):
        """Apply exact first-chunk scores, calibration, and boundary guards."""
        start, end, _ = self._chunks[0]
        if self.exact_first_chunk:
            _, repeat_ids = self._pairs[0]
            self.start_idx, self.end_idx = start, end
            hooks = [
                layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True)
                for layer in model.model.layers
            ]
            try:
                model(input_ids=repeat_ids.to(model.device), past_key_values=self._cache, logits_to_keep=1)
            finally:
                for hook in hooks:
                    hook.remove()
        if self.exact_first_chunk and self.score_calibration and len(self._chunks) > 1:
            self._apply_per_sample_calibration(start, end)
        if self.guard_tokens > 0:
            for start, _, _ in self._chunks:
                self.score_val[..., start : start + self.guard_tokens] = 1.0

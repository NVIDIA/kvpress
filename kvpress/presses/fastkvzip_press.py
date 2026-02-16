# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

import torch
from huggingface_hub import hf_hub_download
from torch import nn
from transformers import AutoConfig, Gemma3ForConditionalGeneration, PreTrainedModel
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from kvpress.presses.base_press import SUPPORTED_MODELS, BasePress

logger = logging.getLogger(__name__)


class FastKVzipGate(nn.Module):
    """
    Fast KVzip gate architecture (https://arxiv.org/abs/2601.17668).
    """

    def __init__(
        self,
        index: int,
        input_dim: int,
        nhead: int,
        ngroup: int,
        dtype: torch.dtype,
        output_dim: int = 16,
        sink: int = 16,
    ):
        super().__init__()
        self.index = index
        self.output_dim = output_dim
        self.nhead = nhead
        self.ngroup = ngroup
        self.sink = sink

        self.q_proj = nn.Linear(input_dim, nhead * ngroup * output_dim, bias=True, dtype=dtype)
        self.k_proj = nn.Linear(input_dim, nhead * output_dim, bias=False, dtype=dtype)
        self.q_norm = Qwen3RMSNorm(output_dim)
        self.k_norm = Qwen3RMSNorm(output_dim)
        self.k_base = nn.Parameter(torch.zeros([nhead, 1, sink, output_dim]))
        self.b = nn.Parameter(torch.zeros([nhead, 1, ngroup], dtype=dtype))

        self.d = math.sqrt(self.output_dim)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.squeeze(0)  # bsz = 1
        nseq = hidden_states.shape[0]  # sequence x dim
        hidden_shape = (nseq, self.nhead, -1, self.output_dim)

        queries = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        keys = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        queries = queries.transpose(0, 1).transpose(-1, -2)
        keys = keys.transpose(0, 1)

        # head x seq x 1 x group
        logit = torch.matmul(keys, queries) / self.d + self.b.unsqueeze(2)
        # head x 1 x sink x group
        logit_base = torch.matmul(self.k_base, queries) / self.d
        score = 1 / (1 + torch.exp(logit_base - logit).sum(2, keepdim=True))

        score = score.mean(-1)  # n_head, seq, 1
        return score.squeeze(-1).unsqueeze(0)  # bsz x n_head x seq

    def extra_repr(self):
        # Customize the print output
        repr_str = f"index={self.index}, output_dim={self.output_dim}, nhead={self.nhead}, ngroup={self.ngroup}\n"
        if self.sink != 0:
            repr_str += f"k_base shape: {self.k_base.shape}\n"
        repr_str += f"b shape: {self.b.shape}\n"
        return repr_str


def load_fastkvzip(model_name: str = "Qwen/Qwen3-8B", device: str = "cuda"):
    """Load trained gate weights"""
    if not model_name:
        raise AssertionError("Model_name is empty. Please check load_gate.")
    state_dict, gate_id = get_gate_weight(model_name)

    dtype = state_dict[0]["q_proj.weight"].dtype
    head_group_outdim, input_dim = state_dict[0]["q_proj.weight"].shape
    head_outdim, _ = state_dict[0]["k_proj.weight"].shape
    output_dim = state_dict[0]["q_norm.weight"].shape[-1]
    nhead = head_outdim // output_dim
    ngroup = head_group_outdim // head_outdim

    m = re.search(r"sink(\d+)", gate_id)
    sink = int(m.group(1)) if m else 0

    modules = []
    for idx, weight in enumerate(state_dict):
        module = FastKVzipGate(idx, input_dim, nhead, ngroup, dtype, output_dim, sink).to(device)
        module.load_state_dict(weight)
        modules.append(module)

    print(f"load gate {gate_id} ({module})")
    return modules


def get_gate_id(model_name: str):
    """Get the gate id from model names"""
    config = AutoConfig.from_pretrained(model_name)
    if hasattr(config, "text_config"):
        config = config.text_config
    ngroup = config.num_attention_heads // config.num_key_value_heads
    file_name = f"q{ngroup}_dim16_sink16"

    model_name = model_name.split("/")[-1].lower()
    gate_id = os.path.join(model_name, file_name + ".pt")
    return gate_id


def get_gate_weight(model_name: str):
    """Load trained gate weights from HuggingFace"""
    gate_id = get_gate_id(model_name)
    file_path = hf_hub_download(repo_id="Jang-Hyun/Fast-KVzip", filename=gate_id, repo_type="model")

    # Load the PyTorch tensor/dictionary
    weights = torch.load(file_path, weights_only=False)["module"]
    return weights, gate_id


@dataclass
class FastKVzipPress(BasePress):
    """
    Fast KVzip estimates KV importance scores using gates trained on KVzip scores.

    In this code, we implement Fast KVzip with minimal changes to this repository.
    For a fully optimized implementation with actual compression and chunked-prefill,
    please refer to the original repository (https://github.com/Janghyun1230/FastKVzip).

    Based on Fast KVzip (https://arxiv.org/abs/2601.17668).
    Authors: Jang-Hyun Kim, Dongyoon Han, Sangdoo Yun
    Affiliation: NAVER AI Lab

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    layerwise : bool, default=False
        Whether to enable uniform compression ratios across layers.
        When False, while the overall KV cache compression ratio is maintained,
        each layer has a different compression ratio.
    n_sink : int, default=4
        Number of initial tokens to preserve as attention sinks.
    window_size : int, default=4096
        Number of tokens in the local window retained during chunked prefilling.
    window_ratio : float, default=0.02
        Fraction of the context length used to calculate the local window size retained during short-context prefilling.
    """

    compression_ratio: float = 0.0
    layerwise: bool = False

    n_sink: int = 4
    window_size: int = 4096  # for chunked prefilling with long contexts
    window_ratio: float = 0.02

    gates: list[nn.Module] | None = field(init=False, default=None)
    score_val: list[torch.Tensor] | torch.Tensor | None = field(init=False, default=None)

    def post_init_from_model(self, model):
        """
        Automatically load gates for the model.
        """
        if self.gates is None:
            try:
                self.gates = load_fastkvzip(model_name=model.config.name_or_path, device=model.device)
            except Exception as e:
                raise RuntimeError(
                    "The gates for the given model are not released! "
                    "Please check the available models at: "
                    "https://huggingface.co/Jang-Hyun/Fast-KVzip/tree/main"
                ) from e

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        """
        Context manager that handles both initial prefilling and Fast KVzip scoring/compression.

        This overrides the base class __call__ method to implement the Fast KVzip algorithm:
        1. First yield: allows initial prefilling with context and KV importance scoring via gates
        2. After yield: performs KV eviction based on the importance scores
        """
        if not isinstance(model, SUPPORTED_MODELS):
            logger.warning(f"Model {type(model)} not tested, supported models: {SUPPORTED_MODELS}")

        self.post_init_from_model(model)
        hooks = []
        try:
            self.score_val = [None for _ in range(len(model.model.layers))]  # reset every prefilling
            language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
            for layer in language_model.layers:
                if isinstance(model, Gemma3ForConditionalGeneration) and layer.self_attn.is_sliding:
                    # Skip layers with sliding window attention, only for Gemma3
                    continue
                layer.self_attn.rotary_emb = language_model.rotary_emb
                hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))
            yield

            self.compress_post(model)  # Perform compression

        finally:
            for hook in hooks:
                hook.remove()

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        Override the forward_hook of BasePress.
        During the forward_hook, Fast KVzip calculates importance scores,
        aggregates scores across all layers, and then performs compression.
        """

        hidden_states = kwargs["hidden_states"]
        q_len = hidden_states.shape[1]

        # Don't compress after pre-filling
        if kwargs["cache_position"][-1] > q_len:
            return output

        self._score_fast(module, hidden_states)
        return output

    def _score_fast(self, module: nn.Module, hidden_states: torch.Tensor):
        """
        Calculate the KV importance scores.
        """
        layer_idx = int(module.layer_idx)

        self.gates[layer_idx] = self.gates[layer_idx].to(hidden_states.device)
        scores = self.gates[layer_idx](hidden_states)
        scores[:, :, : self.n_sink] = 1.0

        ctx_len = scores.size(-1)
        if ctx_len < 32000:
            window_size = int(ctx_len * self.window_ratio)
        else:
            window_size = self.window_size
        scores[:, :, -window_size:] = 1.0

        self.score_val[layer_idx] = scores

    def compress_post(self, model: PreTrainedModel):
        """
        Obtain the indices of KV pairs to be evicted.
        Adopted from adakv_press.compress (fake compression). KVzip does not rely on safeguards.
        """
        self.score_val = torch.stack(self.score_val, dim=0)

        if self.compression_ratio > 0:
            n_layer, bsz, num_key_value_heads, ctx_len = self.score_val.shape

            # calculate the pruned KV pairs across layers
            if self.layerwise:
                nl = int(bsz * num_key_value_heads * ctx_len * self.compression_ratio)
                n_pruned_layers = nl * torch.ones(n_layer, device=self.score_val.device, dtype=torch.int)
            else:
                n_pruned_indices = int(self.score_val.numel() * self.compression_ratio)
                pruned_indices = torch.topk(-self.score_val.reshape(-1), n_pruned_indices).indices
                n_tokens_per_layer = bsz * num_key_value_heads * ctx_len
                n_pruned_layers = torch.bincount(pruned_indices // n_tokens_per_layer, minlength=n_layer).int()

            for layer in model.model.layers:
                module = layer.self_attn
                layer_idx = int(module.layer_idx)

                assert module.config._attn_implementation != "eager", "eager mode not supported"

                scores = self.score_val[layer_idx]

                # Compute bottom-k across heads
                n_pruned = n_pruned_layers[layer_idx].cpu()
                indices = torch.topk(-scores.reshape(bsz, -1), n_pruned, dim=1).indices.flatten().cpu()

                # Save indices to mask during the attention mechanism. Please refer to attention_patch.py for details
                batch_indices = torch.arange(bsz, device=n_pruned.device).repeat_interleave(n_pruned)
                head_indices = indices // ctx_len
                seq_indices = indices % ctx_len
                module.masked_key_indices = (batch_indices, head_indices, seq_indices)

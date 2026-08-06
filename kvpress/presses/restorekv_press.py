# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass, field

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
from transformers import PreTrainedModel

from kvpress.presses.kvzip_press import KVzipPress

logger = logging.getLogger(__name__)


@dataclass
class RestoreKVPress(KVzipPress):
    """RestoreKV (https://arxiv.org/abs/2608.01247): learned restoration on top of KVzip.

    Before eviction, n=8 restore tokens attend to the full KV cache in one
    LoRA-adapted pass, producing a context-conditioned restore cache that fills
    n*L*H slots of the same budget while the base evictor fills the rest
    (budget-matched). LoRA is active only for this one-time pass. Only the restore
    embeddings and LoRA (~0.4%) are trained, by self-distillation from the
    full-cache teacher. Adapters are published as PEFT LoRA adapters at
    `higokri/RestoreKV-<model>[_plus]` for Llama-3.1-8B-Instruct and Qwen3-8B, and
    loaded automatically from the model name.
    """

    restore_embeddings: torch.Tensor | None = field(init=False, default=None, repr=False)
    restore_model_name: str | None = field(init=False, default=None, repr=False)
    encoding_restore_tokens: bool = field(init=False, default=False, repr=False)

    @property
    def num_restore_tokens(self) -> int:
        if self.restore_embeddings is None:
            return 0
        return self.restore_embeddings.shape[0]

    @property
    def adapter_name(self) -> str:
        return "restorekv_plus" if self.kvzip_plus_normalization else "restorekv"

    def post_init_from_model(self, model: PreTrainedModel):
        model_name = model.config.name_or_path.split("/")[-1]
        variant = "_plus" if self.kvzip_plus_normalization else ""
        restore_model_name = f"higokri/RestoreKV-{model_name}{variant}"
        if restore_model_name == self.restore_model_name:
            return

        embeddings_path = hf_hub_download(restore_model_name, "restore_embeddings.safetensors")
        self.restore_embeddings = load_file(embeddings_path)["restore_embeddings"].to(model.device, dtype=model.dtype)
        if self.adapter_name not in getattr(model, "peft_config", {}):
            model.load_adapter(restore_model_name, adapter_name=self.adapter_name)
        model.disable_adapters()
        self.restore_model_name = restore_model_name
        logger.info("Loaded %s with %d restore tokens", restore_model_name, self.num_restore_tokens)

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        # Skip KVzip scoring while the restore tokens are being encoded (see append_restore_tokens).
        if self.encoding_restore_tokens:
            return output
        return super().forward_hook(module, input, kwargs, output)

    def append_restore_tokens(self, model: PreTrainedModel):
        """Single LoRA-adapted restore pass over the full cache (before eviction);
        appends the n restore tokens' K/V in place. Scoring hooks are skipped."""
        cache_position = torch.arange(
            self.context_length, self.context_length + self.num_restore_tokens, device=self.restore_embeddings.device
        )
        self.encoding_restore_tokens = True
        model.set_adapter(self.adapter_name)
        model.enable_adapters()
        try:
            with torch.inference_mode():
                model.model(
                    inputs_embeds=self.restore_embeddings.unsqueeze(0),
                    past_key_values=self._cache,
                    position_ids=cache_position.unsqueeze(0),
                    cache_position=cache_position,
                    use_cache=True,
                )
        finally:
            model.disable_adapters()
            self.encoding_restore_tokens = False

    def compress_post(self, model: PreTrainedModel):
        # KVzip has finished scoring the context; append the restore tokens to the
        # full cache (budget-matched), then let KVzip evict the lowest-scored pairs.
        self.append_restore_tokens(model)
        requested_ratio = self.compression_ratio
        if self.context_length > 0:
            restore_overhead = self.num_restore_tokens / self.context_length
            self.compression_ratio = min(1.0, requested_ratio + restore_overhead)
        try:
            super().compress_post(model)
        finally:
            self.compression_ratio = requested_ratio

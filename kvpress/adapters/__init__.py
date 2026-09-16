# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kvpress.adapters.base import (
    ModelAdapter,
    get_adapter,
    get_adapter_from_model_type,
    get_adapter_from_module,
    has_adapter,
    register_adapter,
)
from kvpress.adapters.llama_like import LlamaLikeAdapter
from kvpress.adapters.qwen3_5 import Qwen3_5Adapter

__all__ = [
    "LlamaLikeAdapter",
    "ModelAdapter",
    "Qwen3_5Adapter",
    "get_adapter",
    "get_adapter_from_model_type",
    "get_adapter_from_module",
    "has_adapter",
    "register_adapter",
]

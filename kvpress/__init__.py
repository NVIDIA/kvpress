# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kvpress.pipeline import KVPressTextGenerationPipeline
from kvpress.presses.base_press import BasePress
from kvpress.presses.composed_press import ComposedPress
from kvpress.presses.expected_attention_press import ExpectedAttentionPress
from kvpress.presses.key_rerotation_press import KeyRerotationPress
from kvpress.presses.knorm_press import KnormPress
from kvpress.presses.observed_attention_press import ObservedAttentionPress
from kvpress.presses.per_layer_compression_press import PerLayerCompressionPress
from kvpress.presses.random_head_press import RandomHeadPress
from kvpress.presses.random_press import RandomPress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.simlayerkv_press import SimLayerKVPress
from kvpress.presses.snapkv_press import SnapKVPress
from kvpress.presses.streaming_llm_press import StreamingLLMPress
from kvpress.presses.think_press import ThinKPress
from kvpress.presses.tova_press import TOVAPress


# Strategy 1: compact but use exec
# import re
# import inspect
# import importlib
# for name in ["llama", "mistral", "phi3", "qwen2"]:
#     module = importlib.import_module(f"transformers.models.{name}.modeling_{name}")
#     attention_classes = getattr(module, f"{name.upper()}_ATTENTION_CLASSES")
#     for key, cls in attention_classes.items():
#         updated_source_code = re.sub(
#             r"cache_kwargs = {(.*?)\}", r'cache_kwargs = {\1, "query_states": query_states}', inspect.getsource(cls)
#         )
#         exec(updated_source_code, module.__dict__)  # security risk here
#         attention_classes[key] = module.__dict__[cls.__name__]

# Strategy 2: cleaner but less compact
from kvpress.models.utils import update_attn_implementations
update_attn_implementations()


__all__ = [
    "BasePress",
    "ComposedPress",
    "ScorerPress",
    "ExpectedAttentionPress",
    "KnormPress",
    "ObservedAttentionPress",
    "RandomHeadPress",
    "RandomPress",
    "SimLayerKVPress",
    "SnapKVPress",
    "StreamingLLMPress",
    "ThinKPress",
    "TOVAPress",
    "KVPressTextGenerationPipeline",
    "PerLayerCompressionPress",
    "KeyRerotationPress",
]

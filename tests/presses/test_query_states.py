# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch
from transformers import Gemma3TextConfig, LlamaConfig, Phi3Config, Qwen3Config
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention
from transformers.models.llama.modeling_llama import LlamaAttention, rotate_half
from transformers.models.phi3.modeling_phi3 import Phi3Attention
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from kvpress import CAMPress, KVzipPress, NonCausalAttnPress, ThinKPress
from kvpress.utils import get_prerope_query_states


@torch.no_grad()
@pytest.mark.parametrize(
    "config_cls,attention_cls",
    [
        (LlamaConfig, LlamaAttention),
        (Phi3Config, Phi3Attention),
        (Qwen3Config, Qwen3Attention),
        (Gemma3TextConfig, Gemma3Attention),
    ],
    ids=["llama", "phi3", "qwen3", "gemma3"],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("float_rope", [False, True], ids=["same_dtype_rope", "float32_rope"])
@pytest.mark.parametrize("caller", ["think_window", "think_short", "kvzip", "cam_prefill", "cam_decode", "non_causal"])
def test_queries_match_before_helper_refactor(config_cls, attention_cls, dtype, float_rope, caller):
    """Check the queries consumed by each press against its original projection/RoPE order."""
    torch.manual_seed(0)
    config = config_cls(hidden_size=32, num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    module = attention_cls(config, layer_idx=0).to(dtype).eval()
    seq_len = 1 if caller == "cam_decode" else 7
    hidden_states = torch.randn(2, seq_len, 32, dtype=dtype)
    # Extra history catches accidental use of the first positions in suffix-aligned callers.
    position_len = seq_len + 5 if caller.startswith("cam") or caller == "non_causal" else seq_len
    angles = torch.randn(2, position_len, 8, dtype=torch.float32 if float_rope else dtype)
    cos, sin = angles.cos(), angles.sin()
    kwargs = {"position_embeddings": (cos, sin)}

    if caller.startswith("think"):
        window_size = 3 if caller == "think_window" else 12
        press = ThinKPress(window_size=window_size)
        queries = get_prerope_query_states(module, hidden_states[:, -window_size:])
        cos, sin = cos[:, -window_size:], sin[:, -window_size:]
    else:
        queries = get_prerope_query_states(module, hidden_states)
        if caller.startswith("cam"):
            queries = queries[:, :, -1:, :]
            cos, sin = cos[:, -1:, :], sin[:, -1:, :]
        elif caller == "non_causal":
            cos, sin = cos[:, -seq_len:, :], sin[:, -seq_len:, :]
    expected = (queries * cos.unsqueeze(1)) + (rotate_half(queries) * sin.unsqueeze(1))
    keys = torch.randn(2, 2, seq_len, 8, dtype=expected.dtype)

    if caller.startswith("think"):
        actual = press.compute_window_queries(module, hidden_states, kwargs["position_embeddings"])
    elif caller == "non_causal":
        press = NonCausalAttnPress(chunk_size=4)
        with patch.object(press, "non_causal_chunked_attn", wraps=press.non_causal_chunked_attn) as attention:
            press.score(module, hidden_states, keys, keys, None, kwargs)
        actual = attention.call_args.args[0]
    else:
        with patch("torch.matmul", wraps=torch.matmul) as matmul:
            if caller.startswith("cam"):
                CAMPress._compute_current_token_attention(module, hidden_states, keys, kwargs)
            else:
                press = KVzipPress(n_sink=1)
                press.start_idx, press.end_idx = 1, 3
                press._compute_cross_attention(module, hidden_states, keys, kwargs)
        actual = matmul.call_args.args[0]
        if caller == "kvzip":
            assert actual.shape == (2, 2, 2, seq_len, 8)
            actual = actual.reshape_as(expected)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

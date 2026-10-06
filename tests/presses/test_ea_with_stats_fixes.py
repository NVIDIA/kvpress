# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from transformers import Qwen2Config

from kvpress import ExpectedAttentionStatsPress
from kvpress.presses import expected_attention_with_stats
from kvpress.presses.expected_attention_with_stats import ExpectedAttentionStats


def fake_model(config, name_or_path):
    config.name_or_path = name_or_path
    return SimpleNamespace(config=config)


def test_ea_stats_of_models_without_head_dim_in_their_config(monkeypatch):
    stats_ids = []
    monkeypatch.setattr(ExpectedAttentionStats, "from_pretrained", lambda stats_id: stats_ids.append(stats_id))
    config = Qwen2Config(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2)
    assert getattr(config, "head_dim", None) is None
    model = fake_model(config, "Qwen/Qwen2.5-0.5B-Instruct")

    ExpectedAttentionStatsPress()._maybe_load_stats_from_hub(model)
    assert stats_ids == ["alessiodevoto/exp_att_stats_Qwen_Qwen2.5-0.5B-Instruct_kmfoda_booksum_100_1000_4"]
    assert expected_attention_with_stats.get_head_dim(config) == 16

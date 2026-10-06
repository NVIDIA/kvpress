# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM, DynamicCache, Qwen2Config

from kvpress import ExpectedAttentionStatsPress
from kvpress.presses import expected_attention_with_stats
from kvpress.presses.expected_attention_with_stats import ExpectedAttentionStats

MODEL_NAME = "MaxJeblick/llama2-0b-unit-test"


def fake_model(config, name_or_path):
    config.name_or_path = name_or_path
    return SimpleNamespace(config=config)


def random_stats(model):
    config = model.config
    stats = ExpectedAttentionStats(
        num_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        head_dim=config.head_dim,
        dataset_name="kmfoda/booksum",
        model_name=config.name_or_path,
        num_samples=100,
        sample_seq_len=1000,
        n_sink=4,
    )
    stats.query_mean.data.normal_()
    stats.query_cov.data = torch.eye(config.head_dim).expand_as(stats.query_cov).clone()
    return stats


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs a second device")
def test_ea_stats_are_moved_to_the_device_of_each_layer(tmp_path):
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval()
    random_stats(model).save_pretrained(tmp_path)
    press = ExpectedAttentionStatsPress(compression_ratio=0.5, stats_folder=str(tmp_path))
    press.post_init_from_model(model)
    # The statistics are loaded on the model device, which is not the one of every layer with a device_map
    press.mu, press.cov = press.mu.cuda(), press.cov.cuda()

    cache = DynamicCache()
    with press(model):
        model(torch.randint(0, 1024, (1, 32)), past_key_values=cache)
    assert cache.get_seq_length() == 16


def test_ea_stats_of_models_without_head_dim_in_their_config(monkeypatch):
    stats_ids = []
    monkeypatch.setattr(ExpectedAttentionStats, "from_pretrained", lambda stats_id: stats_ids.append(stats_id))
    config = Qwen2Config(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2)
    assert getattr(config, "head_dim", None) is None
    model = fake_model(config, "Qwen/Qwen2.5-0.5B-Instruct")

    ExpectedAttentionStatsPress()._maybe_load_stats_from_hub(model)
    assert stats_ids == ["alessiodevoto/exp_att_stats_Qwen_Qwen2.5-0.5B-Instruct_kmfoda_booksum_100_1000_4"]
    assert expected_attention_with_stats.get_head_dim(config) == 16

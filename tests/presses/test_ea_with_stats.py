# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from kvpress.presses.expected_attention_with_stats import ExpectedAttentionStats, ExpectedAttentionStatsPress


def test_load_stats():
    for stats_id in ExpectedAttentionStatsPress.available_stats():
        ExpectedAttentionStats.from_pretrained(stats_id)

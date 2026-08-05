# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from kvpress import RestoreKVPress
from kvpress.presses.kvzip_press import KVzipPress


def test_adapter_name_and_restore_token_count():
    base = RestoreKVPress()
    plus = RestoreKVPress(kvzip_plus_normalization=True)
    assert base.adapter_name == "restorekv"
    assert plus.adapter_name == "restorekv_plus"

    assert base.num_restore_tokens == 0
    base.restore_embeddings = torch.zeros(8, 16)
    assert base.num_restore_tokens == 8


def test_budget_matching_counts_restore_tokens(monkeypatch):
    observed_ratios = []
    monkeypatch.setattr(KVzipPress, "compress_post", lambda self, model: observed_ratios.append(self.compression_ratio))

    press = RestoreKVPress(compression_ratio=0.5)
    press.context_length = 100
    press.restore_embeddings = torch.zeros(8, 4)
    # compress_post appends the restore tokens first; stub that out to isolate the budget logic.
    monkeypatch.setattr(press, "append_restore_tokens", lambda model: None)
    press.compress_post(model=None)

    # 8 restore tokens out of 100 context tokens -> evict 8% extra so the total budget matches.
    assert observed_ratios == [0.58]
    assert press.compression_ratio == 0.5

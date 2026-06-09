# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from kvpress import CapPress
from tests.fixtures import unit_test_model  # noqa: F401


def test_cap_press(unit_test_model):  # noqa: F811
    for press in [
        CapPress(compression_ratio=0.5, tau=5.0),
        CapPress(compression_ratio=0.8, tau=5.0),
        CapPress(compression_ratio=0.8, tau=10.0),
        CapPress(compression_ratio=0.8, tau=5.0, n_sink=8),
    ]:
        with press(unit_test_model):
            input_ids = torch.arange(10, 40).to(unit_test_model.device)
            unit_test_model(input_ids.unsqueeze(0), use_cache=True)

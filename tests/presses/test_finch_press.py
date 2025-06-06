# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from kvpress import FinchPress
from tests.fixtures import unit_test_model  # noqa: F401


def test_finch_press(unit_test_model):  # noqa: F811
    for press in [
        FinchPress(0.5),
        FinchPress(0.5, rerotate_keys=False),
        FinchPress(0.5, normalize_scores=False),
        FinchPress(0.2, chunk_length=5),
    ]:
        with press(unit_test_model):
            bos = unit_test_model.generation_config.bos_token_id
            input_ids = torch.arange(10, 20)
            input_ids[0] = bos
            input_ids[8] = bos
            unit_test_model(input_ids.unsqueeze(0))

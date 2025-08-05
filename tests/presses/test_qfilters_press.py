# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from kvpress.presses.qfilter_press import QFilterPress


def test_load_qfilters():
    for model_name in QFilterPress.available_qfilters():
        QFilterPress.load_q_filters(model_name)

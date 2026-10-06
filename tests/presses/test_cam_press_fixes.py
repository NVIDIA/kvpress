# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from kvpress import AdaKVPress, CAMPress, SnapKVPress


def test_cam_rejects_adakv_base_press():
    with pytest.raises(ValueError, match="requires a ScorerPress"):
        CAMPress(base_press=AdaKVPress(SnapKVPress()))

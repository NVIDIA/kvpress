# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from kvpress import KnormPress, PerLayerCompressionPress
from tests.fixtures import unit_test_model  # noqa: F401


@dataclass
class RecordingKnormPress(KnormPress):
    models: list = field(default_factory=list)

    def post_init_from_model(self, model):
        self.models.append(model)


def test_post_init_from_model_is_forwarded(unit_test_model):  # noqa: F811
    inner_press = RecordingKnormPress()
    with PerLayerCompressionPress(press=inner_press, compression_ratios=[0.5, 0.25])(unit_test_model):
        pass
    assert inner_press.models == [unit_test_model]

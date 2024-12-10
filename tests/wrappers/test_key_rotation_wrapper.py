# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from kvpress import KnormPress
from kvpress.wrappers.rerotate_keys_press import KeyRerotationPress
from tests.fixtures import kv_press_pipeline, unit_test_model  # noqa: F401


def test_presses_run(kv_press_pipeline):  # noqa: F811
    press = KnormPress(compression_ratio=0.5)
    wrapped_press = KeyRerotationPress(press)

    context = "This is a test article. It was written on 2022-01-01."
    questions = ["When was this article written?"]
    answers = kv_press_pipeline(context, questions=questions, press=wrapped_press)["answers"]

    assert len(answers) == 1
    assert isinstance(answers[0], str)

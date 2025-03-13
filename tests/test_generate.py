# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from kvpress import KnormPress
from tests.fixtures import kv_press_unit_test_pipeline  # noqa: F401


def test_generate(kv_press_unit_test_pipeline):  # noqa: F811
    context = "This is a test article. It was written on 2022-01-01."
    press = KnormPress(compression_ratio=0.4)

    # Answer with pipeline
    pipe_answer = kv_press_unit_test_pipeline(context, press=press, max_new_tokens=10)["answer"]

    # Answer with model.generate
    context += "\n"  # kv press pipeline automatically adds a newline if no chat template
    model = kv_press_unit_test_pipeline.model
    tokenizer = kv_press_unit_test_pipeline.tokenizer
    with press(model):
        inputs = tokenizer(context, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        generate_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generate_answer = generate_answer[len(context) :]

    assert pipe_answer == generate_answer

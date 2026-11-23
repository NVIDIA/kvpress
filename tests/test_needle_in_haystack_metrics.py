# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest


def test_needle_metrics_use_prediction_as_hypothesis():
    pytest.importorskip("rouge")
    from evaluation.benchmarks.needle_in_haystack.calculate_metrics import calculate_metrics

    results = pd.DataFrame({"needle": ["Paris is the capital of France"], "predicted_answer": ["Paris is"]})

    (score,) = calculate_metrics(results)

    assert score["rouge-1"]["p"] == pytest.approx(1.0)
    assert score["rouge-1"]["r"] == pytest.approx(2 / 6)
    assert score["rouge-2"]["p"] == pytest.approx(1.0)
    assert score["rouge-2"]["r"] == pytest.approx(1 / 5)
    assert score["rouge-l"]["p"] == pytest.approx(1.0)
    assert score["rouge-l"]["r"] == pytest.approx(2 / 6)

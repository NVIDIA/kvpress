# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from evaluation.benchmarks.loogle.calculate_metrics import get_exact_match


def test_unparsed_prediction_keeps_the_reference_key_count():
    reference = "{'a': 1, 'b': 2, 'c': 3, 'd': 4}"

    assert get_exact_match(reference, "nope") == (0, 4)
    assert get_exact_match(reference, "[]") == (0, 4)
    assert get_exact_match(reference, reference) == (4, 4)
    assert get_exact_match("nope", reference) == (0, 1)

    matched = get_exact_match(reference, reference)
    missed = get_exact_match(reference, "nope")
    assert (matched[0] + missed[0]) / (matched[1] + missed[1]) == 0.5

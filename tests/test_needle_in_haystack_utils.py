# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pandas as pd
import pytest

from evaluation.benchmarks.needle_in_haystack.utils import insert_needle_in_haystack


@pytest.mark.parametrize("index", [[17, 23], [17, 0]])
def test_insert_needle_uses_first_sampled_row(index):
    data = pd.DataFrame(
        {
            "context": ["before after", "different context"],
            "needle": ["needle", "other"],
            "question": ["Where?", "Why?"],
            "answer_prefix": ["Here:", "Because:"],
            "max_new_tokens": [8, 16],
        },
        index=index,
    )
    tokenizer = SimpleNamespace(
        encode=lambda text, **kwargs: text.split(),
        decode=lambda tokens, **kwargs: " ".join(tokens),
    )

    result = insert_needle_in_haystack(
        data, tokenizer, max_context_length=154, needle_depth=50, context_wrapper="{context}"
    )

    assert result.to_dict("records") == [
        {
            "context": "before needle after",
            "needle": "needle",
            "needle_depth": 50,
            "question": "Where?",
            "answer_prefix": "Here:",
            "max_new_tokens": 8,
        }
    ]

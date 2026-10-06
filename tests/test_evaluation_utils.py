# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

from evaluation.benchmarks.aime25.calculate_metrics import score_aime as score_aime25
from evaluation.benchmarks.infinite_bench.calculate_metrics import calculate_metrics as infinite_bench_scorer
from evaluation.benchmarks.math500.calculate_metrics import score_aime as score_math500
from evaluation.benchmarks.needle_in_haystack.utils import insert_needle_in_haystack
from evaluation.benchmarks.utils import extract_boxed

CODE_DEBUG_QUESTION = (
    "Which funtion has deliberate error?\n\nOptions:\nA. Resource.pkgname\nB. repack_carchive\nC. cmd_gen\nD. _init"
)
LONGBOOK_CHOICE_QUESTION = (
    "Which of the following is NOT one of Alain's chores at Hall Farm?\n\nOnly one of the following options is "
    "correct, tell me the answer using one single letter (A, B, C, or D). Don't say anything else.\n"
    "A. Walking Georgie\nB. Taking care of Totty\nC. Working in the dairy\nD. Light housework"
)


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


def haystack_df(index):
    return pd.DataFrame(
        {
            "context": [" ".join(f"word{i}" for i in range(1000))],
            "needle": ["The secret is 42."],
            "question": ["What is the secret?"],
            "answer_prefix": ["Answer:"],
            "max_new_tokens": [10],
        },
        index=[index],
    )


def infinite_bench_df(task, question, answer, prediction, context="Some long context."):
    # Same layout as load_dataset("MaxJeblick/InfiniteBench", data_dir=task, split="test").to_pandas()
    return pd.DataFrame(
        {
            "context": [context],
            "question": [question],
            "answer": [np.array(answer, dtype=object)],
            "task": [task],
            "predicted_answer": [prediction],
        }
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"Answer: \boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        (r"Answer: \boxed{\left\{x \mid x > 0\right\}}", r"\left\{x \mid x > 0\right\}"),
    ],
)
def test_extract_boxed_handles_nested_latex(text, expected):
    assert extract_boxed(text) == expected


def test_extract_boxed_selects_first_or_last_answer():
    text = r"Draft: \boxed{1}. Final: \boxed{2}."

    assert extract_boxed(text) == "1"
    assert extract_boxed(text, last=True) == "2"


@pytest.mark.parametrize(
    "text",
    [
        "No boxed answer",
        r"Incomplete: \boxed{\frac{1}{2}",
    ],
)
def test_extract_boxed_returns_none_without_complete_box(text):
    assert extract_boxed(text) is None


def test_aime25_scores_last_boxed_answer():
    assert score_aime25(r"Draft: \boxed{1}. Final: \boxed{\frac{1}{2}}.", r"\frac{1}{2}")


def test_math500_scores_first_boxed_answer():
    assert score_math500(r"First: \boxed{\frac{1}{2}}. Later: \boxed{1}.", r"\frac{1}{2}")


# task: (question, answer, perfect prediction)
INFINITE_BENCH_ROWS = {
    "passkey": ("What is the pass key?", ["71432"], "71432"),
    "number_string": ("What is the sequence of digits?", ["2200012222"], "2200012222"),
    "kv_retrieval": ('\nKey: "798c2306-5ad1-42a9"\n ', ["5e6b7b90-710d-4953"], "5e6b7b90-710d-4953"),
    "code_run": ("Please give me the exact number of the return value of func_6577(-7).", ["-14"], "-14"),
    "code_debug": (CODE_DEBUG_QUESTION, ["B"], "B"),
    "math_find": ("The largest number of the list is: ", ["88"], "88"),
    "math_calc": ("What's the intermediate results of the given numerical expression?", ["79", "5", "73"], "79, 5, 73"),
    "longbook_qa_eng": ("Which among Annalisa, Seb and Peyton is not Mrs. Bronwyn's child?", ['"Peyton"'], "Peyton"),
    "longbook_qa_chn": ("云景, 素素, 云林哪一个人物是第二个登场？", ["素素"], "素素"),
    "longbook_choice_eng": (LONGBOOK_CHOICE_QUESTION, ["A"], "A"),
    "longdialogue_qa_eng": ("Which character is $$MASK$$ ?", ["ACE", "ACE ROTHSTEIN"], "ACE"),
}


def test_insert_needle_in_haystack_reads_first_row_by_position():
    df = insert_needle_in_haystack(haystack_df(index=7), WordTokenizer(), max_context_length=400, needle_depth=[0, 50])

    assert df["needle_depth"].tolist() == [0, 50]
    assert all("The secret is 42." in context for context in df["context"])
    assert df["question"].tolist() == ["What is the secret?"] * 2


def test_insert_needle_in_haystack_rejects_too_small_max_context_length():
    with pytest.raises(ValueError, match="max_context_length"):
        insert_needle_in_haystack(haystack_df(index=0), WordTokenizer(), max_context_length=150, needle_depth=50)


@pytest.mark.filterwarnings("error::DeprecationWarning")
@pytest.mark.parametrize("task", list(INFINITE_BENCH_ROWS))
def test_infinite_bench_scores_perfect_prediction(task):
    question, answer, prediction = INFINITE_BENCH_ROWS[task]
    assert infinite_bench_scorer(infinite_bench_df(task, question, answer, prediction)) == 1.0


@pytest.mark.parametrize("query_aware", [False, True])
@pytest.mark.parametrize(
    ("task", "correct_option", "wrong_option"),
    [("code_debug", "repack_carchive", "cmd_gen"), ("longbook_choice_eng", "Walking Georgie", "Light housework")],
)
def test_infinite_bench_choice_label_includes_option_text(task, correct_option, wrong_option, query_aware):
    question, answer, _ = INFINITE_BENCH_ROWS[task]
    context = "Some long context."
    if query_aware:
        context, question = context + question, ""

    assert infinite_bench_scorer(infinite_bench_df(task, question, answer, correct_option, context)) == 1.0
    assert infinite_bench_scorer(infinite_bench_df(task, question, answer, wrong_option, context)) == 0.0

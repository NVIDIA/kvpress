# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import re

import pandas as pd


def string_match_part(preds, refs):
    score = (
        sum([max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) for pred, ref in zip(preds, refs)])
        / len(preds)
        * 100
    )
    return round(score, 2)


def string_match_all(preds, refs):
    score = (
        sum(
            [sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref) for pred, ref in zip(preds, refs)]
        )
        / len(preds)
        * 100
    )
    return round(score, 2)


def calculate_metrics(df: pd.DataFrame) -> dict:
    scores = {}

    np_pattern = re.compile(r"[\x00-\x1f]")
    df["predicted_answer"] = df["predicted_answer"].apply(lambda x: np_pattern.sub("", x.strip()).strip())

    for task, df_task in df.groupby("task"):
        task_category = task.split("_")[0]
        metric_fn = string_match_part if task_category == "qa" else string_match_all
        preds = df_task["predicted_answer"].tolist()
        refs = df_task["answer"].tolist()
        score = metric_fn(preds, refs)
        scores[task] = {"string_match": score}
    return scores

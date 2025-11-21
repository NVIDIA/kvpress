# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def score(predicted_answer, expected_answer):
    # From https://github.com/THUDM/LongBench/blob/main/pred.py (extract_answer function)
    predicted_answer = predicted_answer.replace("*", "")
    r1 = f"The correct answer is ({expected_answer})" in predicted_answer
    r2 = f"The correct answer is {expected_answer}" in predicted_answer
    return r1 or r2


def calculate_metrics(df):
    df["score"] = df.apply(lambda row: score(row["predicted_answer"], row["answer"]), axis=1)
    metrics = {"average": df["score"].mean()}
    metrics.update(df.groupby("difficulty")["score"].mean())
    metrics.update(df.groupby("length")["score"].mean())
    return metrics

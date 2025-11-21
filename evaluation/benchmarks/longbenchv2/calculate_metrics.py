# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def score(predicted_answer, expected_answer):
    # From https://github.com/THUDM/LongBench/blob/main/pred.py (extract_answer function)
    predicted_answer = predicted_answer.replace("*", "").strip()
    r1 = predicted_answer == f"The correct answer is ({expected_answer})"
    r2 = predicted_answer == f"The correct answer is {expected_answer}"
    return r1 or r2


def calculate_metrics(df):
    return df.apply(lambda row: score(row["predicted_answer"], row["answer"]), axis=1).mean()

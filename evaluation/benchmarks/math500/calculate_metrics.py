# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

from ..utils import extract_boxed


def score_aime(pred_answer, true_answer):
    return extract_boxed(pred_answer) == str(true_answer)


def calculate_metrics(df: pd.DataFrame) -> dict:
    correct = 0
    answered = 0
    for index, row in df.iterrows():
        correct += score_aime(row["predicted_answer"], row["answer"])
        answered += "boxed{" in row["predicted_answer"]
    return {"correct": correct, "answered": answered, "accuracy": correct / len(df), "total": len(df)}

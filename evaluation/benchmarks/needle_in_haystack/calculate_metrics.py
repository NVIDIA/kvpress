# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from rouge import Rouge

scorer = Rouge()


def calculate_metrics(df: pd.DataFrame) -> list[dict]:
    scores = []
    for index, row in df.iterrows():
        score = scorer.get_scores(row["needle"].strip(), row["predicted_answer"].strip())[0]
        scores.append(score)
    return scores

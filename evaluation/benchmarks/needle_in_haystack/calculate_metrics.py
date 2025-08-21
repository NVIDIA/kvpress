# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

def calculate_metrics(df: pd.DataFrame) -> dict:
    scores = []
    for index, row in df.iterrows():
        score = scorer.score(row["needle"], row["predicted_answer"])["rouge1"].fmeasure * 10
        scores.append(score)
    return {"rouge1": sum(scores) / len(scores)}
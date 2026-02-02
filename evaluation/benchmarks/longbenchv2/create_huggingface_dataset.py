# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datasets import Dataset, load_dataset

# Templates from https://github.com/THUDM/LongBench/blob/main/prompts/0shot.txt
context_template = """Please read the following text and answer the question below.
<text>
{context}
</text>

"""

question_template = """What is the correct answer to this question: {question}
Choices:
(A) {A}
(B) {B}
(C) {C}
(D) {D}

Format your response as follows: "The correct answer is (insert answer here)."""

# Longbench-v2
df = load_dataset("THUDM/LongBench-v2", split="train").to_pandas()
df["context"] = df["context"].apply(lambda x: context_template.format(context=x))
df["question"] = df.apply(
    lambda row: question_template.format(
        question=row["question"],
        A=row["choice_A"],
        B=row["choice_B"],
        C=row["choice_C"],
        D=row["choice_D"],
    ),
    axis=1,
)
df["max_new_tokens"] = 16
df["answer_prefix"] = ""
Dataset.from_pandas(df).push_to_hub("simonjegou/LongBench-v2", config_name="0shot", split="test")

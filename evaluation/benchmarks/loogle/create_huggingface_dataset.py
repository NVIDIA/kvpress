# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import json

from datasets import Dataset, load_dataset

# Templates based on https://github.com/bigai-nlco/LooGLE/blob/main/config/task2prompt.json
context_prompt = {
    "shortdep_qa": "Please answer the question based on the long texts below. \n{input}",
    "longdep_qa": "Please answer the question based on the long texts below. \n{input}",
    "shortdep_cloze": "Please fill in the clozes based on the given long texts below. Each of the placeholder '<mask-n>' in the question could be an entity of Person, Location or Organiocation. The same masks represent the same entity. Output a json format answer, for example: {{'<mask-0>': 'Bob', '<mask-1>': 'Gorrosion Magazine','<mask-2>': 'Bethel Horizon'}}\n{input}",  # noqa
    "longdep_summarization": "Please generate a summary of the below paper. \n{input}",
}

question_prompt = {
    "shortdep_qa": "\nQuestion: {Q}\n",
    "longdep_qa": "\nQuestion: {Q}\n",
    "shortdep_cloze": "\n Question: {Q} What are the masked entities?\n",
    "longdep_summarization": "{Q}\n",
}

answer_prefix = {
    "shortdep_qa": "Answer: ",
    "longdep_qa": "Answer: ",
    "shortdep_cloze": "Answer:",
    "longdep_summarization": "Summarization: ",
}

# Source: https://github.com/bigai-nlco/LooGLE/blob/main/config/task2maxlen.json
max_new_tokens = {"shortdep_qa": 300, "longdep_qa": 500, "longdep_summarization": 500, "shortdep_cloze": 50}

for task in ["shortdep_qa", "longdep_qa", "shortdep_cloze", "longdep_summarization"]:

    source_task = "summarization" if task == "longdep_summarization" else task
    df = load_dataset("bigainlco/LooGLE", source_task, split="test").to_pandas()

    if task == "shortdep_cloze":
        df["answer"] = df["answer"].apply(
            lambda values: json.dumps(
                {f"<mask-{i}>": value for i, value in enumerate(values)}, ensure_ascii=False
            )
        )

    df["context"] = df["context"].apply(lambda x: context_prompt[task].format(input=x))
    df["question"] = df["question"].apply(lambda x: question_prompt[task].format(Q=x))
    df["answer_prefix"] = answer_prefix[task]
    df = df[["context", "question", "answer_prefix", "answer"]]
    df["task"] = task
    df["max_new_tokens"] = max_new_tokens[task]

    # Push to hub
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub("simonjegou/loogle", config_name=task, split="test")

# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import re
from pathlib import Path

import pandas as pd
from datasets import Dataset

# Source: https://github.com/hsiehjackson/RULER/blob/main/scripts/data/synthetic/constants.py

QUESTION_PATTERNS = {
    "niah": re.compile(r"What (?:is|are all) the special magic"),
    "vt": re.compile(r"Question: Find all variables that are assigned the value"),
    "cwe": re.compile(r"Question: What are the 10 most common words in the above list\?"),
    "fwe": re.compile(r"Question: Do not provide any explanation\."),
    "qa": re.compile(r"Answer the question based on the given documents\."),
}

ANSWER_PATTERNS = {
    "niah": re.compile(r"The special magic"),
    "vt": re.compile(r"Answer:"),
    "cwe": re.compile(r"Answer:"),
    "fwe": re.compile(r"Answer:"),
    "qa": re.compile(r"Answer:"),
}

# Source: https://github.com/hsiehjackson/RULER/blob/main/scripts/data/synthetic/constants.py
MAX_NEW_TOKENS = {
    "niah": 128,
    "vt": 30,
    "cwe": 120,
    "fwe": 50,
    "qa": 32,
}


def get_dataframe(path):
    """
    Parse the data from the provided path and return a DataFrame with the context, question, answers and task
    """

    assert re.match(r".*\/\d+$", str(path)), "The path should must ends with the context length (e.g. with /4096)"

    df_list = []
    for task_path in Path(path).glob("**/*.jsonl"):

        # Load dataframe
        df = pd.read_json(task_path, lines=True)
        task = task_path.parent.stem
        question_pattern = QUESTION_PATTERNS[task.split("_")[0]]
        answer_pattern = ANSWER_PATTERNS[task.split("_")[0]]

        # Split the context and the question based on the pattern
        def split_context_question(text):
            idx = list(question_pattern.finditer(text))[-1].start()
            context, qa = text[:idx], text[idx:]
            idx = answer_pattern.search(qa).start()
            question, answer = qa[:idx], qa[idx:]
            return context, question, answer

        df["context"], df["question"], df["answer_prefix"] = zip(*df["input"].apply(split_context_question))
        df["task"] = task
        df["max_new_tokens"] = MAX_NEW_TOKENS[task.split("_")[0]]
        df_list.append(df)

    # Concatenate all the dataframes
    df = pd.concat(df_list)
    df = df[["context", "question", "answer_prefix", "outputs", "task", "max_new_tokens"]]
    df = df.rename(columns={"outputs": "answer"}).reset_index(drop=True)

    return df


if __name__ == "__main__":
    data_dir = Path("/mnt/workspace/projects/RULER/scripts/data/data/")  # output of the generate.sh script
    repo_id = "simonjegou/ruler"

    # Loop over all the context lengths
    for path in data_dir.glob("*/"):
        context_length = path.stem
        print(f"Processing context length {context_length}")
        df = get_dataframe(path)
        dataset = Dataset.from_pandas(df)
        dataset.push_to_hub(repo_id=repo_id, config_name=context_length, split="test")

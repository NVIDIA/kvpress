# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from datasets import Dataset, load_dataset

# yarn_mistral_templates from: https://github.com/THUDM/LongBench/blob/main/pred.py
context_prefix = {
    "0shot": "Please read the following text and answer the question below.\n\n<text>\n{context}\n</text>\n\n",
    "cot": "Please read the following text and answer the question below.\n\n<text>\n{context}\n</text>\n\n",
    "rag": "Please read the following retrieved text chunks and answer the question below.\n\n<text>\n{context}\n</text>\n\n",
    "nocontext": "",
}

question_template = {
    "0shot": "What is the correct answer to this question: {question}\nChoices:\n(A) {C_A}\n(B) {C_B}\n(C) {C_C}\n(D) {C_D}\n\n",
    "cot": "What is the correct answer to this question: {question}\nChoices:\n(A) {C_A}\n(B) {C_B}\n(C) {C_C}\n(D) {C_D}\n\n",
    "rag": "What is the correct answer to this question: {question}\nChoices:\n(A) {C_A}\n(B) {C_B}\n(C) {C_C}\n(D) {C_D}\n\n",
    "nocontext": "What is the correct answer to this question: {question}\nChoices:\n(A) {C_A}\n(B) {C_B}\n(C) {C_C}\n(D) {C_D}\n\n",
}

answer_prefix = {
    "0shot": 'Format your response as follows: "The correct answer is (insert answer here)".',
    "cot": 'Format your response as follows: "The correct answer is (insert answer here)".\n\nLet’s think step by step:',
    "rag": 'Format your response as follows: "The correct answer is (insert answer here)".',
    "nocontext": 'What is the single, most likely answer choice? Format your response as follows: "The correct answer is (insert answer here)".',
}

DATA_NAME_TO_MAX_NEW_TOKENS = {"0shot": 128, "cot": 1024}

# Longbench-v2
for task in ["0shot", "cot"]:
    dataset = load_dataset("THUDM/LongBench-v2", split="train")
    dataset = dataset.map(lambda x: {"context": context_prefix[task].format(context=x["context"].strip())})
    dataset = dataset.map(
        lambda x: {
            "question": question_template[task].format(
                question=x["question"].strip(),
                C_A=x["choice_A"].strip(),
                C_B=x["choice_B"].strip(),
                C_C=x["choice_C"].strip(),
                C_D=x["choice_D"].strip(),
            )
        }
    )
    df = dataset.to_pandas()
    df["answer_prefix"] = answer_prefix.get(task, "")
    # df = df[["context", "question", "answer_prefix", "answers", "all_classes"]]
    df["task"] = task
    # be a bit more generous with token generation to avoid any cut-offs
    df["max_new_tokens"] = DATA_NAME_TO_MAX_NEW_TOKENS[task] + 20

    # Push to hub
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub("Xnhyacinth/LongBench-v2", config_name=task, split="test")

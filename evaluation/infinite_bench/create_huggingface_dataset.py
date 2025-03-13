# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import re

from datasets import Dataset, Features, Sequence, Value, load_dataset

"""
| Task Name            | Context       | # Examples | Avg Input Tokens | Avg Output Tokens | Description                                                                                 |
| -------------------- | ------------- | ---------- | ---------------- | ----------------- | ------------------------------------------------------------------------------------------- |
| passkey              | Synthetic     | 590        | 122.4k           | 2.0               | Retrieving hidden keys in a noisy long context.                                             |
| kv_retrieval         | Synthetic     | 500        | 89.9k            | 22.7              | Finding the corresponding value from a dictionary and a key.                                |
| number_string        | Synthetic     | 590        | 122.4k           | 4.0               | Locating repeated hidden numbers in a noisy long context.                                   |
| code_run             | Synthetic     | 400        | 75.2k            | 1.3               | Simulating execution of multiple simple, synthetic functions.                               |
| code_debug           | Code Document | 394        | 114.7k           | 4.8               | Finding which function in a code repo contains an crashing error (in multiple choice form). |
| math_find            | Synthetic     | 350        | 87.9k            | 1.3               | Finding special integers in a lengthy list.                                                 |
| longbook_qa_eng      | Fake Book     | 351        | 192.6k           | 4.8               | Free-form question answering based on the fake book.                                        |
| longdialogue_qa_eng  | Script        | 200        | 103.6k           | 3.4               | Identification of talkers in partially anonymized scripts.                                  |
| longbook_choice_eng  | Fake Book     | 229        | 184.4k           | 5.3               | Multiple choice questions derived from the fake book.                                       |
"""

"""
Examples:
passkey:
    context: "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there. The pass key is 71432. Remember it. 71432 is the pass key. The grass is green. The sky is blue."
    question: "What is the pass key?"
    answer: ["71432"]

kv_retrieval:
    context: "Extract the value corresponding to the specified key in the JSON object below. JSON data: {"e6aa4656-0eb5-4e1d-ad33-1ded282e0a78" ..."
    question: "Key: "ce06788c-71b4-4f7a-b196-0fd9965a59c5" The value associated with the specified key is:"
    answer: ["00a00042-6bcb-494f-9c35-57180f1e7251"]

number_string:
    context: "There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there. The sequence of digits is 2200012222. Remember it. 2200012222 is the sequence of digits. The grass is green. The sky is blue."
    question: "What is the sequence number?"
    answer: ["2200012222"]

longdialogue_qa_eng:
    context: "Below is a dialogue script where one random occurrence of a character name is replaced with "$$MASK$$", and you should try to guess who that character is. The dialogue: --- BEASTS OF THE SOUTHERN WILD Written by Lucy Alibar & Benh Zeitlin FINAL DRAFT: Based on the stage play "Juicy and Delicious"
    question: "Which character is $$MASK$$ ?"
    answer: [ "ACE", "ACE ROTHSTEIN" ]

longbook_qa_eng:
    context: "Read the book below and answer a question. ‘Yes, of course, if it’s fine to-morrow,’ said Mrs Bronwyn. ‘But you’ll have to be up with the lark,’ she added. "
    question: "Which among Annalisa, Seb, Peyton, and Gannonmarie is not Mrs. Bronwyn's child?"
    answer: [ "\"Peyton\"" ]

longbook_choice_eng:
    context: "Read the book and answer the question. With a single drop of ink for a mirror, the Egyptian sorcerer undertakes to reveal to any chance comer far-reaching visions of the past. This is what I undertake to do for you, reader. With this drop of ink at the end of my pen, I will show you the roomy workshop "
    question: "Which of the following is NOT one of Alain's chores at Hall Farm? Only one of the following options is correct, tell me the answer using one single letter (A, B, C, or D). Don't say anything else. A. Walking Georgie B. Taking care of Totty C. Working in the dairy D. Light housework"
    answer: "["A"]"
"""

ft = Features(
    {
        "id": Value("int64"),
        "context": Value("string"),
        "input": Value("string"),
        "answer": Sequence(Value("string")),
        "options": Sequence(Value("string")),
    }
)

# yarn_mistral_templates from: https://github.com/OpenBMB/InfiniteBench/blob/main/src/prompt.py
context_prefix = {
    "passkey": "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n\n{context}\n\n",
    "kv_retrieval": "Extract the value corresponding to the specified key in the JSON object below.\n\n{context}\n\n",
    "number_string": "There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{context}\n\n",
    "longdialogue_qa_eng": 'Below is a dialogue script where one random occurrence of a character name is replaced with "$$MASK$$", and you should try to guess who that character is.\n\n{context}\n\n',
    "longbook_qa_eng": "Read the book below and answer a question. Be very concise in your answer.\n\n{context}\n\n",
    "longbook_qa_chn": "请根据以下书籍回答我的问题。\n\n{context}\n\n",
    "longbook_choice_eng": "Read the book and answer the question.\n\n{context}\n\n",
    "math_calc": "Let us calculate the intermediate values of an expression.\n\nExpression: 1 + 3 + 4\nValues: [1, 4, 8]\n\nExpression: 8 - 3 + 2 - 4\nValues: [8, 5, 7, 3]\n\nExpression: {context}\n\n",
    "code_debug": "Following is a Python code where exactly one of the functions/methods has a deliberate error that makes it crash.\n\n{context}\n\n",
    "longbook_sum_eng": "Summarize the book below:\n\n{context}\n\n",
    "math_find": "{prefix}\n\n{context}\n\n{input}\n\n",
    "code_run": "There is a function called {func} in the following Python code.\n\n{context}\n\n",
}
question_template = {
    "longbook_choice_eng": "\n\nOnly one of the following options is correct, tell me the answer using one single letter (A, B, C, or D). Don't say anything else.\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}",
    "code_debug": "\n\nOptions:\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}",
}

answer_prefix = {
    "kv_retrieval": "The value associated with the specified key is",
    "passkey": "The pass key is:",
    "number_string": "The sequence of digits is",
    "longbook_sum_eng": "Summary:",
    "longbook_choice_eng": "The letter of the correct answer is",
    "longbook_qa_eng": "Answer:",
    "longbook_qa_chn": "答案：",
    "math_calc": "Values:",
    "code_run": "The return value is:",
    "code_debug": "The correct option is:",
    "longdialogue_qa_eng": "The name that has been replaced with $$MASK$$ is likely",
}
DATA_NAME_TO_MAX_NEW_TOKENS = {
    "passkey": 6,
    "number_string": 12,
    "kv_retrieval": 50,
    "longbook_sum_eng": 1200,
    "longbook_choice_eng": 40,
    "longbook_qa_eng": 40,
    "longbook_qa_chn": 40,
    "longdialogue_qa_eng": 40,
    "math_find": 3,
    "math_calc": 30000,
    "code_run": 5,
    "code_debug": 5,
}

for task in [
    "passkey",
    "kv_retrieval",
    "number_string",
    "longdialogue_qa_eng",
    "longbook_qa_eng",
    "longbook_choice_eng",
    "code_run",
    "code_debug",
    "math_find",
    "math_calc",
    "longbook_sum_eng",
    "longbook_qa_chn",
]:
    dataset = load_dataset("xinrongzhang2022/InfiniteBench", features=ft)
    df = dataset[task].to_pandas()
    assert (df.columns == ["id", "context", "input", "answer", "options"]).all()
    if task not in ["longbook_choice_eng", "code_debug"]:
        # Only longbook_choice_eng and code_debug have non-empty options column
        assert (df["options"].apply(len) == 0).all()

    df = df.rename(columns={"input": "question"})
    df["answer_prefix"] = answer_prefix.get(task, "")
    if task == "math_find":
        # https://github.com/OpenBMB/InfiniteBench/blob/main/src/eval_utils.py#L328C1-L340C1
        def update_math_find_context(row):
            prompt = row["question"]
            context = row["context"]
            find_result = re.findall(r"The .+ of", prompt)

            if not find_result:
                raise AssertionError()

            target_number = find_result[0].lower()[:-3]
            prefix = f"What is {target_number} in the following list?"

            return context_prefix["math_find"].format(
                prefix=prefix,
                context=context,
                input=prompt,
            )

        df["context"] = df.apply(update_math_find_context, axis=1)
        df["context"] = df["context"].apply(
            lambda x: x.replace(
                "You should answer with only one number, no other words. The largest number of the list is:", ""
            )
        )
    elif task == "code_run":
        # https://github.com/OpenBMB/InfiniteBench/blob/main/src/eval_utils.py#L272
        def update_context(row):
            find_result = re.findall(r"func_[0-9]+\(\-?[0-9]+\)", row["question"])

            if not find_result:
                raise AssertionError()

            func_call = find_result[0]
            func = func_call.split("(")[0]

            return context_prefix["code_run"].format(
                func=func,
                context=row["context"],
            )

        df["context"] = df.apply(update_context, axis=1)
    else:
        df["context"] = df["context"].apply(lambda x: context_prefix[task].format(context=x))

    if task in ["longbook_choice_eng", "code_debug"]:
        df["question"] = df["question"] + df["options"].apply(
            lambda x: question_template[task].format(OPTION_A=x[0], OPTION_B=x[1], OPTION_C=x[2], OPTION_D=x[3])
        )
        df["answer"] = df.apply(lambda row: ["ABCD"[list(row.options).index(row.answer)]], axis=1)

    if task == "kv_retrieval":
        # moved to answer prefix
        df["question"] = df["question"].apply(
            lambda x: x.replace("The value associated with the specified key is:", "")
        )

    df = df[["context", "question", "answer_prefix", "answer"]]
    df["task"] = task

    # be a bit more generous with token generation to avoid any cut-offs
    df["max_new_tokens"] = DATA_NAME_TO_MAX_NEW_TOKENS[task] + 20

    # Push to hub
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub("MaxJeblick/InfiniteBench", config_name=task, split="test")

# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from datasets import Dataset, load_dataset

# yarn_mistral_templates from: https://github.com/THUDM/LongBench/blob/main/LongBench/pred.py
context_prefix = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\n",
    "qasper": 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\n',
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\n",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\n",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\n",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\n",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\n",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\n",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\n",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\n",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n",
    "lcc": "Please complete the code given below. \n{context}",
    "repobench-p": "Please complete the code given below. \n{context}",
}

question_template = {
    "narrativeqa": "Question: {input}\n\n",
    "qasper": "Question: {input}\n\n",
    "multifieldqa_en": "Question: {input}\n",
    "multifieldqa_zh": "问题：{input}\n",
    "hotpotqa": "Question: {input}\n",
    "2wikimqa": "Question: {input}\n",
    "musique": "Question: {input}\n",
    "dureader": "问题：{input}\n",
    "gov_report": "Now, write a one-page summary of the report.\n\n",
    "qmsum": "Query: {input}\n",
    "multi_news": "Now, write a one-page summary of all the news.\n\n",
    "vcsum": "",
    "trec": "{input}",
    "triviaqa": "{input}",
    "samsum": "{input}",
    "lsht": "{input}",
    "passage_count": "Please enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\n",
    "passage_retrieval_en": '{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\n',
    "passage_retrieval_zh": '{input}\n\n请输入摘要所属段落的编号。答案格式必须是"段落1"，"段落2"等格式\n\n',
    "lcc": "{input}",
    "repobench-p": "{input}",
}

answer_prefix = {
    "narrativeqa": "Answer:",
    "qasper": "Answer:",
    "multifieldqa_en": "Answer:",
    "multifieldqa_zh": "回答：",
    "hotpotqa": "Answer:",
    "2wikimqa": "Answer:",
    "musique": "Answer:",
    "dureader": "回答：",
    "gov_report": "Summary:",
    "qmsum": "Answer:",
    "trec": "Type:",
    "multi_news": "Summary:",
    "samsum": "Summary:",
    "triviaqa": "Answer:",
    "vcsum": "会议总结：",
    "passage_count": "The final answer is: ",
    "passage_retrieval_en": "The answer is: ",
    "passage_retrieval_zh": "答案是：",
    "lcc": "Next line of code:\n",
    "repobench-p": "Next line of code:\n",
}
DATA_NAME_TO_MAX_NEW_TOKENS = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64,
}

# Longbench
for task in [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]:
    dataset = load_dataset("THUDM/LongBench", task, split="test")
    dataset = dataset.map(lambda x: {"context": context_prefix[task].format(**x)})

    if task == "trec":
        dataset = dataset.map(
            lambda x: {"input": question_template[task].format(input=x["input"].removesuffix("Type:"))}
        )
    elif task == "triviaqa":
        dataset = dataset.map(
            lambda x: {"input": question_template[task].format(input=x["input"].removesuffix("Answer:"))}
        )
    elif task == "samsum":
        dataset = dataset.map(
            lambda x: {"input": question_template[task].format(input=x["input"].removesuffix("Summary:"))}
        )
    else:
        dataset = dataset.map(lambda x: {"input": question_template[task].format(**x)})

    df = dataset.to_pandas()
    df = df.rename(columns={"input": "question"})
    df["answer_prefix"] = answer_prefix.get(task, "")
    # df = df[["context", "question", "answer_prefix", "answers", "all_classes"]]
    df["task"] = task

    # be a bit more generous with token generation to avoid any cut-offs
    df["max_new_tokens"] = DATA_NAME_TO_MAX_NEW_TOKENS[task] + 20

    # Push to hub
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub("Xnhyacinth/LongBench", config_name=task, split="test")

# Longbench-e
for task in [
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]:
    dataset = load_dataset("THUDM/LongBench", f"{task}_e", split="test")
    dataset = dataset.map(lambda x: {"context": context_prefix[task].format(**x)})

    if task == "trec":
        dataset = dataset.map(
            lambda x: {"input": question_template[task].format(input=x["input"].removesuffix("Type:"))}
        )
    elif task == "triviaqa":
        dataset = dataset.map(
            lambda x: {"input": question_template[task].format(input=x["input"].removesuffix("Answer:"))}
        )
    elif task == "samsum":
        dataset = dataset.map(
            lambda x: {"input": question_template[task].format(input=x["input"].removesuffix("Summary:"))}
        )
    else:
        dataset = dataset.map(lambda x: {"input": question_template[task].format(**x)})

    df = dataset.to_pandas()
    df = df.rename(columns={"input": "question"})
    df["answer_prefix"] = answer_prefix.get(task, "")
    # df = df[["context", "question", "answer_prefix", "answers", "all_classes"]]
    df["task"] = task

    # be a bit more generous with token generation to avoid any cut-offs
    df["max_new_tokens"] = DATA_NAME_TO_MAX_NEW_TOKENS[task] + 20

    # Push to hub
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub("Xnhyacinth/LongBench", config_name=f"{task}_e", split="test")

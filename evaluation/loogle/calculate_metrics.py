# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import nltk
import pandas as pd
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge


# Code below is adapted from https://github.com/bigai-nlco/LooGLE/blob/main/Evaluation/automatic_metrics.py
def get_bleu_score(reference, hypothesis):
    reference, hypothesis = (
        reference.replace("\n", " ").split(),
        hypothesis.replace("\n", " ").split(),
    )

    bleu1 = sentence_bleu([reference], hypothesis, weights=(1, 0, 0, 0))
    bleu4 = sentence_bleu([reference], hypothesis, weights=(0, 0, 0, 1))
    return {"bleu1": bleu1, "bleu4": bleu4}


def get_rouge_score(reference, hypothesis, metric="r"):
    rouge = Rouge()
    rouge_ = rouge.get_scores(hyps=[hypothesis], refs=[reference])[0]
    return dict((key, rouge_[key][metric]) for key in ["rouge-1", "rouge-2", "rouge-l"])


def get_meteor_score(reference, hypothesis):
    reference, hypothesis = (
        reference.replace("\n", " ").split(),
        hypothesis.replace("\n", " ").split(),
    )
    meteor = single_meteor_score(set(reference), set(hypothesis))
    return {"meteor": float(meteor)}


def get_exact_match(reference, hypothesis):
    try:
        reference = eval(reference)
        count = len(reference)
        hypothesis = eval(hypothesis)
        assert isinstance(hypothesis, dict)
    except Exception:
        return 0, 1

    exact_score_count = 0
    for key, value in reference.items():
        if hypothesis.get(key) == value:
            exact_score_count += 1
    return exact_score_count, count


def get_partial_match(reference, hypothesis):
    reference = eval(reference)
    count = len(reference)
    try:
        hypothesis = eval(hypothesis)
        assert isinstance(hypothesis, dict)
        partial_score_count = 0
        for key in reference:
            if key in hypothesis:
                true_set = set(reference[key].split())
                pred_set = set(hypothesis[key].split())
                if len(true_set.intersection(pred_set)) > 0:
                    partial_score_count += 1
        return partial_score_count, count
    except Exception:
        return 0, count


def try_except_metric(metric_fn):
    def wrapped_metric(answer, predicted_answer):
        try:
            return metric_fn(answer, predicted_answer)
        except Exception as e:
            print(f"Cannot calculate metric: {e}" f" on answer:{answer} and predicted_answer:{predicted_answer}")
            return {key: 0.0 for key in metric_fn("Hi there", "hi there")}

    return wrapped_metric


def calculate_metrics(df: pd.DataFrame) -> dict:
    nltk.download("wordnet")

    scores: dict = {}
    for task, df_task in df.groupby("task"):
        scores[task] = {}
        if task == "shortdep_cloze":
            for prefix, metric_fn in [
                ("exact", get_exact_match),
                ("partial", get_partial_match),
            ]:
                match, count = zip(*df_task.apply(lambda x: metric_fn(x["answer"], x["predicted_answer"]), axis=1))
                scores[task][f"{prefix}_match"] = round(sum(match) / sum(count), 4)

        else:
            df["predicted_answer"] = df["predicted_answer"].apply(lambda x: x if isinstance(x, str) else "<NONE>")

            for metric_fn in [get_bleu_score, get_rouge_score, get_meteor_score]:  # type: ignore
                metric_fn = try_except_metric(metric_fn)

                metric_scores = [metric_fn(row["answer"], row["predicted_answer"]) for _, row in df_task.iterrows()]
                scores[task].update(pd.DataFrame(metric_scores).mean().to_dict())

            # BERT scores (batched)
            scores[task]["bert"] = (
                score(df_task["answer"].to_list(), df_task["predicted_answer"].to_list(), lang="EN")[1].mean().item()
            )

    return scores

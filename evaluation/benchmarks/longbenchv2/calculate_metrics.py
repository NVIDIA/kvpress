# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re


def extract_answer(response):
    response = response.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if match:
        return match.group(1)
    else:
        match = re.search(r"The correct answer is ([A-D])", response)
        if match:
            return match.group(1)
        else:
            return None


def calculate_metrics(df):
    predictions = df["predicted_answer"].tolist()
    answers = df["answer"].tolist()
    lengths = df["length"].tolist()
    difficulties = df["difficulty"].tolist()
    return scorer(predictions, answers, lengths, difficulties)


def scorer(predictions, answers, lengths, difficulties):
    compensated = False
    easy, hard, short, medium, long = 0, 0, 0, 0, 0
    easy_acc, hard_acc, short_acc, medium_acc, long_acc = 0, 0, 0, 0, 0
    for pred, answer, length, difficulty in zip(predictions, answers, lengths, difficulties):
        acc = int(extract_answer(pred) == answer)
        if compensated and pred["pred"] is None:
            acc = 0.25  # type:ignore[assignment]
        if difficulty == "easy":
            easy += 1
            easy_acc += acc
        else:
            hard += 1
            hard_acc += acc

        if length == "short":
            short += 1
            short_acc += acc
        elif length == "medium":
            medium += 1
            medium_acc += acc
        else:
            long += 1
            long_acc += acc
    scores = ["Overall\tEasy\tHard\tShort\tMedium\tLong"]
    scores.append(
        str(round(100 * (easy_acc + hard_acc) / len(predictions), 1))
        + "\t"
        + str(round(100 * easy_acc / easy, 1))
        + "\t"
        + str(round(100 * hard_acc / hard, 1))
        + "\t"
        + str(round(100 * short_acc / short, 1))
        + "\t"
        + str(round(100 * medium_acc / medium, 1))
        + "\t"
        + str(round(100 * long_acc / long, 1))
    )
    return scores

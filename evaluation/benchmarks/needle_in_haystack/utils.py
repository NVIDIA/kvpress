# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

import pandas as pd
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def insert_needle_in_haystack(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    max_context_length: int,
    needle_depth: int | list[int],
    context_wrapper: str = "This is a very long story book: <book> {context} </book>.",
    needle_text: Optional[str] = None,
    answer_prefix: Optional[str] = None,
    question_text: Optional[str] = None,
) -> pd.DataFrame:
    """
    Inserts the "needle" string into the "context" of each row in the DataFrame at specified depths.
    A new row is created for each depth, and the DataFrame is returned with these new rows.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing at least the columns "context" and "needle".
    tokenizer : PreTrainedTokenizer
        The tokenizer used to encode and decode the context and needle.
    max_context_length : int
        The maximum allowed length (in tokens) for the context, including the needle.
    needle_depths : int | list[int]
        A list of percentages (0-100) indicating how deep into the context the needle should be inserted.
    needle_text : Optional[str]
        The text to insert as the needle. If None, the first row's "needle" column is used.
    answer_prefix : Optional[str]
        The prefix to add to the answer. If None, the first row's "answer_prefix" column is used.
    question_text : Optional[str]
        The text to insert as the question. If None, the first row's "question" column is used.
    max_new_tokens : int
        The maximum number of new tokens to generate. If None, the first row's "max_new_tokens" column is used.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the "context" column modified to include the needle, with a new row
        for each specified depth.
    """

    # Store the original context and needle to be reused for each depth
    original_context = df["context"][0]
    needle_text = needle_text or df["needle"][0]
    question_text = question_text or df["question"][0]
    answer_prefix = answer_prefix or df["answer_prefix"][0]
    max_new_tokens = df["max_new_tokens"][0]

    logger.info(f"Preparing dataset for inference. Needle: {needle_text}")
    tokenized_needle = tokenizer.encode(needle_text, add_special_tokens=False)
    # Account for system prompts and other overhead
    context_length_limit = max_context_length - len(tokenized_needle) - 150
    # Tokenize the original context once
    tokenized_context = tokenizer.encode(original_context, add_special_tokens=False)[:context_length_limit]
    # Initialize a list to hold the new rows
    new_rows = []
    needle_depth = [needle_depth] if isinstance(needle_depth, int) else needle_depth

    for depth in needle_depth:
        # Calculate the insertion index based on the current depth
        needle_index = int(len(tokenized_context) * depth / 100)
        # Create a new tokenized context with the needle inserted
        new_tokenized_context = tokenized_context[:needle_index] + tokenized_needle + tokenized_context[needle_index:]
        decoded_context = tokenizer.decode(new_tokenized_context, skip_special_tokens=True)
        final_context = context_wrapper.format(context=decoded_context)
        new_row = {
            "context": final_context,
            "needle": needle_text,
            "needle_depth": depth,
            "question": question_text,
            "answer_prefix": answer_prefix,
            "max_new_tokens": max_new_tokens,
        }
        new_rows.append(new_row)

    # Create the new DataFrame from the list of rows
    result_df = pd.DataFrame(new_rows)

    return result_df

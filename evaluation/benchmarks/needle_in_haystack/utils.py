# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pandas as pd
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def insert_needle_in_haystack(
    df: pd.DataFrame, tokenizer: PreTrainedTokenizer, max_context_length: int, needle_depth: int
) -> pd.DataFrame:
    """
    Inserts the "needle" string into the "context" of each row in the DataFrame at a specified depth.
    Adapted from the original implementation: https://github.com/gkamradt/LLMTest_NeedleInAHaystack

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing at least the columns "context" and "needle".
    tokenizer : PreTrainedTokenizer
        The tokenizer used to encode and decode the context and needle.
    max_context_length : int
        The maximum allowed length (in tokens) for the context, including the needle.
    needle_depth : int
        The percentage (0-100) indicating how deep into the context the needle should be inserted.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the "context" column modified to include the needle at the specified depth.
    """
    logger.info(f"Preparing dataset for inference with needle in haystack. Needle: {df['needle'][0]}")
    tokenized_needle = tokenizer.encode(df["needle"][0], add_special_tokens=False)
    context_length = max_context_length - len(tokenized_needle) - 150  # account for system prompts
    needle_index = int(context_length * needle_depth / 100)
    # tokenize the context
    df["context"] = df["context"].apply(lambda x: tokenizer.encode(x, add_special_tokens=False)[:context_length])
    # insert the needle at the depth specified in the config
    df["context"] = df["context"].apply(lambda x: x[:needle_index] + tokenized_needle + x[needle_index:])
    # detokenize the context
    df["context"] = (
        "This is a very long story book: <book> "
        + df["context"].apply(lambda x: tokenizer.decode(x, skip_special_tokens=True))
        + " </book>."
    )
    return df

# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data collection utilities for KVzap training.

This module provides functions to:
1. Load and preprocess the Nemotron dataset
2. Tokenize prompts with the KVzip repeat method
3. Extract KVzip+ scores from a model using forward hooks
"""

import pandas as pd
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.models.llama.modeling_llama import repeat_kv
from transformers.integrations.finegrained_fp8 import FP8Linear


def load_nemotron_dataset(
    tokenizer: PreTrainedTokenizerBase,
    min_tokens: int = 750,
    max_tokens: int = 1250,
    n_train_per_subset: int = 500,
    n_test_per_subset: int = 5,
) -> pd.DataFrame:
    """
    Load and preprocess the Nemotron dataset for KVzap training.

    The function:
    1. Loads the nvidia/Nemotron-Pretraining-Dataset-sample dataset (multilingual and multi-domain)
    2. Filters samples to keep only those with sequence length in [min_tokens, max_tokens]
       (ensures uniform sequence length so attention weight denominators aren't influenced by length)
    3. Splits into train/test with balanced sampling across subsets

    Parameters
    ----------
    tokenizer : AutoTokenizer
        Tokenizer to use for computing sequence lengths
    min_tokens : int, optional
        Minimum number of tokens per sample, by default 750
    max_tokens : int, optional
        Maximum number of tokens per sample, by default 1250
    n_train_per_subset : int, optional
        Maximum training samples per subset, by default 500
    n_test_per_subset : int, optional
        Maximum test samples per subset, by default 5

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: text, length, subset, split
    """
    subsets = [
        "Nemotron-CC-MATH",
        "Nemotron-CC-High-Quality",
        "Nemotron-CC-High-Quality-Synthetic",
        "Nemotron-CC-Diverse-QA",
        "Nemotron-CC-Translated-Diverse-QA",
        "Nemotron-Synthetic-Code",
        "Nemotron-SFT-Code",
        "Nemotron-SFT-General",
        "Nemotron-SFT-MATH",
    ]

    # 1. Load all subsets and concatenate them
    df_list = []
    for subset in tqdm(subsets, desc="Loading subsets"):
        df = load_dataset("nvidia/Nemotron-Pretraining-Dataset-sample", subset, split="train").to_pandas()
        df["length"] = df["text"].apply(lambda x: len(tokenizer.encode(x)))
        df["subset"] = subset
        df_list.append(df)
    df = pd.concat(df_list)

    # 2. Remove the samples that are too short or too long
    sub_df = df[(max_tokens > df["length"]) & (df["length"] > min_tokens)]

    # 3. Split into train and test
    df_test = sub_df.groupby("subset").head(n_test_per_subset)
    df_test["split"] = "test"
    df_train = sub_df.drop(df_test.index).groupby("subset").head(n_train_per_subset)
    df_train["split"] = "train"
    df = pd.concat([df_test, df_train]).reset_index(drop=True)

    return df


def repeat_prompt_tokenization(
    tokenizer: PreTrainedTokenizerBase, prompt: str
) -> tuple[torch.Tensor, int, int, int, int]:
    """
    Tokenize a prompt using the KVzip repeat method.

    Builds an extended prompt following the KVzip methodology:
    ```
    user: <prompt>

    Repeat the previous context exactly.
    assistant: <prompt>
    ```

    Parameters
    ----------
    tokenizer : AutoTokenizer
        Tokenizer to use
    prompt : str
        The input prompt text

    Returns
    -------
    tuple[torch.Tensor, int, int, int, int]
        - input_ids: Tokenized input tensor
        - start_prompt: Start index of the original prompt
        - end_prompt: End index of the original prompt
        - start_repeated_prompt: Start index of the repeated prompt
        - end_repeated_prompt: End index of the repeated prompt
    """
    # Repeat the prompt using the chat template
    prompt = prompt.strip()
    messages = [
        {"role": "user", "content": prompt + "\n\nRepeat the previous context exactly."},
        {"role": "assistant", "content": prompt},
    ]

    # Tokenize
    prompt_with_repeat = tokenizer.apply_chat_template(messages, tokenize=False)
    outputs = tokenizer(prompt_with_repeat, return_tensors="pt", return_offsets_mapping=True)

    # Get the start and end indexes of the prompt and the repeated prompt
    # The tokenizer might add newlines at the beginning and end of the prompt
    prefix, repeat, _ = prompt_with_repeat.split(prompt)
    m = outputs.offset_mapping[0, :, 0]
    m = torch.cat([m, torch.tensor([len(prompt_with_repeat)])])
    start_prompt = int(torch.where(m >= len(prefix))[0][0].item())
    end_prompt = int(torch.where(m >= len(prefix) + len(prompt))[0][0].item())
    start_repeated_prompt = int(torch.where(m >= len(prefix) + len(prompt) + len(repeat))[0][0].item())
    end_repeated_prompt = int(torch.where(m >= len(prefix) + 2 * len(prompt) + len(repeat))[0][0].item())

    return outputs.input_ids, start_prompt, end_prompt, start_repeated_prompt, end_repeated_prompt


class KVzapDataCollector:
    """
    Collects KVzip+ importance scores from a language model using forward hooks.


    Parameters
    ----------
    model : AutoModelForCausalLM
        The language model to extract scores from
    tokenizer : AutoTokenizer
        Tokenizer matching the model

    Example
    -------
    >>> collector = KVzapDataCollector(model, tokenizer)
    >>> X, y = collector.collect(df, n_tokens=500)
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        self.model = model
        self.tokenizer = tokenizer

        # Hook communication state (replaces global variables)
        self._data: list = []
        self._start_prompt: int = 0
        self._end_prompt: int = 0
        self._start_repeated_prompt: int = 0
        self._end_repeated_prompt: int = 0

    def _forward_hook(self, module, input, kwargs, output):
        """
        Forward hook to extract KVzip+ scores from the extended prompt.

        This hook computes importance scores for each key-value pair based on:
        1. Attention weights from repeated prompt tokens to original prompt tokens
        2. Normalized by hidden state norms
        3. Weighted by output projection norms

        Results are stored in self._data as tuples of (hidden_states, scores).
        """
        # Get variables
        hidden_states = kwargs["hidden_states"]
        values = kwargs["past_key_values"].layers[module.layer_idx].values
        attn_weights = output[1]

        # Initialize scores with attention weights
        scores = attn_weights

        # Divide by ||h|| (by row)
        h_norm = torch.norm(hidden_states, dim=-1)
        scores = torch.einsum("b h t i, b t -> b h t i", scores, 1 / h_norm)

        # Multiply by ||WoV|| (by column)
        Wo = module.o_proj.weight.transpose(0, 1)
        V = repeat_kv(values, module.num_key_value_groups)
        if isinstance(module.o_proj, FP8Linear):
            scale = module.o_proj.weight_scale_inv.to(V.dtype).transpose(0, 1)
            scale = scale.repeat_interleave(module.o_proj.block_size[0], dim=0)
            scale = scale.repeat_interleave(module.o_proj.block_size[1], dim=1)
            Wo = Wo.to(V.dtype) * scale
        Wo = Wo.view(module.config.num_attention_heads, module.head_dim, module.config.hidden_size)
        WoV_norm = torch.einsum("h i j, b h t i -> b h t j", Wo.to(dtype=V.dtype), V).norm(dim=-1)
        scores = torch.einsum("b h t i, b h i -> b h t i", scores, WoV_norm)

        # Get max for each prompt across the repeated prompt tokens and the KV groups
        scores = scores[
            :, :, self._start_repeated_prompt : self._end_repeated_prompt, self._start_prompt : self._end_prompt
        ].amax(dim=2)
        scores = scores.view(
            scores.shape[0], module.config.num_key_value_heads, module.num_key_value_groups, scores.shape[2]
        ).amax(dim=2)

        # Apply log
        scores = torch.log(scores)

        # Store the results
        self._data.append((hidden_states[0, self._start_prompt : self._end_prompt, :].cpu(), scores[0].T.cpu()))

        return output

    def _register_hooks(self) -> list:
        """
        Register forward hooks on all attention layers to extract KVzip+ scores.

        Returns
        -------
        list
            List of hook handles (can be used to remove hooks later)
        """
        handles = []
        for layer in self.model.model.layers:  # type: ignore[attr-defined]
            handle = layer.self_attn.register_forward_hook(self._forward_hook, with_kwargs=True)
            handles.append(handle)
        return handles

    def collect(self, df: pd.DataFrame, n_tokens: int = 500) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Collect training data by extracting KVzip+ scores from text samples.

        For each text sample in the dataset, this function:
        1. Applies the KVzip repeat prompt method
        2. Runs a forward pass to extract attention-based importance scores
        3. Randomly samples n_tokens tokens per sample

        Parameters
        ----------
        df : pd.DataFrame
            Dataset with a "text" column containing the samples
        n_tokens : int, optional
            Number of tokens to sample per text sample, by default 500

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - X: Hidden states tensor of shape (n_samples * n_tokens, n_layers, hidden_size)
            - y: Score tensor of shape (n_samples * n_tokens, n_layers, n_kv_heads)
        """
        # Register hooks
        handles = self._register_hooks()

        try:
            config = self.model.model.config  # type: ignore[attr-defined]
            n_layers = config.num_hidden_layers
            X = torch.zeros(len(df) * n_tokens, n_layers, config.hidden_size, dtype=self.model.dtype)
            y = torch.zeros(len(df) * n_tokens, n_layers, config.num_key_value_heads, dtype=self.model.dtype)

            for i, text in tqdm(enumerate(df["text"]), total=len(df), desc="Extracting scores"):
                # Get the scores using the repeat prompt method
                tokens, self._start_prompt, self._end_prompt, self._start_repeated_prompt, self._end_repeated_prompt = (
                    repeat_prompt_tokenization(self.tokenizer, text)
                )
                self._data = []
                with torch.no_grad():
                    self.model.model(tokens.to(self.model.device))  # type: ignore[attr-defined]

                # Sample n_tokens tokens randomly
                mask = torch.randperm(len(self._data[0][0]))[:n_tokens]
                for layer_idx, (X_, y_) in enumerate(self._data):
                    X[i * n_tokens : (i + 1) * n_tokens, layer_idx] = X_[mask]
                    y[i * n_tokens : (i + 1) * n_tokens, layer_idx] = y_[mask]

            return X, y
        finally:
            # Clean up hooks
            for handle in handles:
                handle.remove()

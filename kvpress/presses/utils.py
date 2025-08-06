# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
from contextlib import contextmanager

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel


@contextmanager
def patch_rotary_embedding(model):
    """
    A context manager to dynamically patch the `apply_rotary_pos_emb` function
    for any supported model architecture. It captures the query states after
    rotary embeddings are applied.

    Args:
        model (PreTrainedModel): The transformer model instance.

    Yields:
        list: A list that will be populated with the captured query tensors.
    """
    # Dynamically find the model's specific "modeling" module
    try:
        module_path = model.__class__.__module__
        modeling_module = importlib.import_module(module_path)
    except Exception as e:
        raise RuntimeError(f"Failed to import module for {model.__class__.__name__}: {e}")

    # Check for the target function and save the original
    target_function = "apply_rotary_pos_emb"
    if not hasattr(modeling_module, target_function):
        raise AttributeError(
            f"Model architecture '{model.config.model_type}' is not supported. "
            f"The module '{module_path}' does not contain '{target_function}'."
        )

    original_function = getattr(modeling_module, target_function)

    captured_tensors = []

    # Define the new wrapper function
    def patched_function(*args, **kwargs):
        # The original function returns a tuple (query_embed, key_embed)
        q_embed, k_embed = original_function(*args, **kwargs)
        # Capture the query tensor after RoPE is applied
        captured_tensors.append(q_embed.detach())

        return q_embed, k_embed

    # Apply the patch
    setattr(modeling_module, target_function, patched_function)

    try:
        # Yield the list to the user to collect the results
        yield captured_tensors
    finally:
        # Restore the original function once the 'with' block is exited
        setattr(modeling_module, target_function, original_function)


@torch.inference_mode()
def collect_queries(
    model: PreTrainedModel, num_samples: int, q_len: int, n_sink: int, return_stats: bool = False
) -> list[torch.Tensor]:
    """
    Collects query representations from a transformer model using a calibration dataset.

    This function runs the model on a small number of samples from the "kmfoda/booksum" dataset,
    capturing the query tensors after rotary positional embeddings are applied. It trims the
    input text to a maximum length (`q_len`), skips the first `n_sink` tokens (to avoid outliers),
    and returns the collected queries.

    Args:
        model (PreTrainedModel): The transformer model instance.
        num_samples (int): Number of samples to use from the calibration dataset.
        q_len (int): Maximum sequence length to consider for each sample.
        n_sink (int): Number of initial tokens to exclude from the collected queries.
        return_stats (bool): Whether to return the mean and covariance of the queries.

    Returns:
        list or tuple:
            collected_queries (list): List of query tensors, each of shape (num_layers, num_heads, seq_len, head_dim)
            mean_query (torch.Tensor): Mean query vector for each layer and head.
            cov_query (torch.Tensor): Covariance matrix of queries for each layer and head.
            If return_stats is False, only the list of query tensors is returned.
    """

    # Load dataset and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
    dataset = load_dataset("kmfoda/booksum", split=f"train[:{num_samples}]")

    # Cut to max q_len
    dataset = dataset.map(lambda x: {"chapter": x["chapter"][:q_len]})

    collected_queries = []
    for text in tqdm(dataset["chapter"], desc="Collecting queries"):
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with patch_rotary_embedding(model) as captured_queries:
            model(**inputs)
        collected_queries.append(torch.cat(captured_queries, dim=0)[:, :, n_sink:, :])

    if return_stats:
        cat_queries = torch.cat(collected_queries, dim=-2)
        mean_query = cat_queries.mean(dim=-2)
        # compute covariance manually
        centered_queries = cat_queries - mean_query.unsqueeze(-2)
        N = cat_queries.shape[-2]
        cov_query = (centered_queries.transpose(-2, -1) @ centered_queries) / (N - 1)
        return collected_queries, mean_query, cov_query
    else:
        return collected_queries

# example cov for one layer
# q = torch.matmul(h, Wq.T).view(bsz, h.shape[1], n, d)
# cov = torch.einsum("bsni,bsnj->bnij", q, q) / h.shape[1]


def compute_query_covariance(queries: list[torch.Tensor]) -> torch.Tensor:
    """
    Computes the covariance matrix of query representations across a calibration dataset.

    This function takes a list of query tensors (one per sample), where each tensor has shape
    (num_layers, num_heads, seq_len, head_dim). For each layer and head, it computes the empirical
    covariance matrix of the query vectors across the sequence dimension (seq_len), for each sample.
    The resulting covariance matrices are then averaged across all samples to obtain a single
    covariance tensor of shape (num_layers, num_heads, head_dim, head_dim).

    Args:
        queries (list of torch.Tensor): List of query tensors, each of shape
            (num_layers, num_heads, seq_len, head_dim), typically collected from multiple samples.

    Returns:
        torch.Tensor: Covariance tensor of shape (num_layers, num_heads, head_dim, head_dim),
            representing the average covariance of queries for each layer and head.
            The covariance is averaged over the batch dimension.
    """
    covariances = []
    for b in range(len(queries)):
        # Get the queries for current sample
        batch_covs = []
        sample_queries = queries[b]  # (num_layers, num_heads, seq_len, head_dim)
        # Compute the covariance for each head
        for l in range(sample_queries.shape[0]):
            layer_covs = []
            for h in range(sample_queries.shape[1]):
                # Queries[b, h] has shape (seq_len, head_dim)
                # torch.cov expects features as rows, so transpose
                cov = torch.cov(sample_queries[l, h].T)  # (head_dim, head_dim)
                layer_covs.append(cov)
            batch_covs.append(torch.stack(layer_covs))
        covariances.append(torch.stack(batch_covs))
    # We average over the batch dimension
    return torch.stack(covariances).mean(dim=0)

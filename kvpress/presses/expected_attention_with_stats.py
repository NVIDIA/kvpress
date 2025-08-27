# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from kvpress.presses.expected_attention_press import ExpectedAttentionPress


@dataclass
class ExpectedAttentionStatsPress(ExpectedAttentionPress):
    """
    Expected attention press that automatically loads pre-computed query statistics.


    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    n_future_positions : int, default=512
        Number of future positions to consider when computing expected attention.
    n_sink : int, default=4
        Number of initial tokens to exclude from compression (sink tokens).
    use_covariance : bool, default=True
        Whether to include covariance information in expected attention computation.
    use_vnorm : bool, default=True
        Whether to rescale scores using value vector norms.
    epsilon : float, default=0.0
        Small constant added to scores before value norm rescaling.
    stats_dataset : str, default="kmfoda/booksum"
        Dataset used to compute the statistics.
    n_samples : int, default=100
        Number of samples used to compute the statistics.
    sample_seq_len : int, default=1000
        Sequence length used to compute the statistics.
    """

    # Override parent defaults to enable stats by default
    sample_seq_len: int = 1000
    n_samples: int = 100
    stats_dataset: str = "kmfoda/booksum"
    stats_folder: Optional[str] = None

    mu: torch.Tensor = field(init=False, default=None)  # initialized in __post_init_from_model__
    cov: torch.Tensor = field(init=False, default=None)  # initialized in __post_init_from_model__

    def get_query_statistics(self, module: nn.Module, hidden_states: torch.Tensor):
        """
        Override the parent method to use the pre-computed query statistics.
        """
        mu, cov = self.apply_avg_rope(module, self.mu, self.cov, hidden_states.shape[1])
        return mu, cov

    def __post_init_from_model__(self, model):
        """
        Automatically load or compute query statistics for the model.
        """
        if self.stats_folder is not None:
            stats = ExpectedAttentionStats.from_pretrained(self.stats_folder)
        else:
            stats = self._maybe_load_stats_from_hub(model)
        self.mu = stats.query_mean.data.to(model.device)
        self.cov = stats.query_cov.data.to(model.device)

    def _maybe_load_stats_from_hub(self, model: PreTrainedModel):
        """Load statistics from the Hugging Face Hub."""
        stats_id = ExpectedAttentionStats(
            model_name=model.config.name_or_path,
            num_layers=model.config.num_hidden_layers,
            num_heads=model.config.num_attention_heads,
            head_dim=model.config.head_dim,
            dataset_name=self.stats_dataset,
            num_samples=self.n_samples,
            sample_seq_len=self.sample_seq_len,
            n_sink=self.n_sink,
        ).stats_id()
        try:
            return ExpectedAttentionStats.from_pretrained(stats_id)
        except ValueError:
            raise ValueError(
                f"No statistics found for model {stats_id} on the Hub. Please compute them first. "
                "You can do so by running the following code: "
                "```"
                "python expected_attention_with_stats.py --model_name <model_name>"
                "```"
            )

    @contextmanager
    def __call__(self, model):
        self.__post_init_from_model__(model)
        with super().__call__(model):
            yield


class ExpectedAttentionStats(torch.nn.Module, PyTorchModelHubMixin):
    """
    Module that stores the mean and covariance matrix of the queries, possibly uploaded to the HF hub.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dataset_name: str,
        model_name: str,
        num_samples: int,
        sample_seq_len: int,
        n_sink: int,
    ):
        super().__init__()
        self.query_mean = torch.nn.Parameter(torch.zeros(num_layers, num_heads, head_dim))
        self.query_cov = torch.nn.Parameter(torch.zeros(num_layers, num_heads, head_dim, head_dim))
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.num_samples = num_samples
        self.sample_seq_len = sample_seq_len
        self.n_sink = n_sink

    def stats_id(self) -> str:
        """Generate the statistics ID for the model and configuration."""
        return f"alessiodevoto/exp_att_stats_{self.model_name.replace('/', '_')}_{self.dataset_name.replace('/', '_')}_{self.num_samples}_{self.sample_seq_len}_{self.n_sink}"  # noqa: E501


# The code below is used to collect statistics on a dataset.


@contextmanager
def patch_rotary_embedding(model):
    """
    A context manager to dynamically patch the `apply_rotary_pos_emb` function
    for any supported model architecture. It captures the query states before
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

    def patched_function(q_embed, k_embed, *args, **kwargs):
        # Capture the query tensor before RoPE is applied
        captured_tensors.append(q_embed.detach())
        q_embed, k_embed = original_function(q_embed, k_embed, *args, **kwargs)
        return q_embed, k_embed

    # Apply the patch
    setattr(modeling_module, target_function, patched_function)

    try:
        yield captured_tensors
    finally:
        setattr(modeling_module, target_function, original_function)


@torch.inference_mode()
def collect_queries(
    model: PreTrainedModel,
    dataset_name: str,
    num_samples: int,
    q_len: int,
    n_sink: int,
    return_stats: bool = False,
    text_column: str = "chapter",
) -> list[torch.Tensor] | tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Collects query representations from a transformer model using a calibration dataset.

    This function runs the model on a small number of samples from the "kmfoda/booksum" dataset,
    capturing the query tensors after rotary positional embeddings are applied. It trims the
    input text to a maximum length (`q_len`), skips the first `n_sink` tokens (to avoid outliers),
    and returns the collected queries.

    Args:
        model (PreTrainedModel): The transformer model instance.
        dataset_name (str): Name of the dataset to use for collecting statistics.
        num_samples (int): Number of samples to use from the calibration dataset.
        q_len (int): Maximum sequence length to consider for each sample.
        n_sink (int): Number of initial tokens to exclude from the collected queries.
        return_stats (bool): Whether to return the mean and covariance of the queries.
        text_column (str): Name of the column in the dataset containing the text to tokenize.

    Returns:
        list or tuple:
            collected_queries (list): List of query tensors, each of shape (num_layers, num_heads, seq_len, head_dim)
            mean_query (torch.Tensor): Mean query vector for each layer and head.
            cov_query (torch.Tensor): Covariance matrix of queries for each layer and head.
            If return_stats is False, only the list of query tensors is returned.
    """

    # Load dataset and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
    dataset = load_dataset(dataset_name, split=f"train[:{num_samples}]")

    # Cut to max q_len
    dataset = dataset.map(lambda x: {text_column: x[text_column][:q_len]})

    collected_queries = []
    for text in tqdm(dataset[text_column], desc="Collecting queries"):
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--output_path", type=str, default=".")
    parser.add_argument("--dataset_name", type=str, default="kmfoda/booksum")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--sample_seq_len", type=int, default=1000)
    parser.add_argument("--n_sink", type=int, default=4)
    parser.add_argument("--text_column", type=str, default="chapter")
    parser.add_argument("--device_map", type=str, default="auto")
    args = parser.parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map=args.device_map, torch_dtype=torch.bfloat16
    ).eval()
    _, mu, cov = collect_queries(
        model, args.dataset_name, args.num_samples, args.sample_seq_len, args.n_sink, return_stats=True
    )

    stats = ExpectedAttentionStats(
        num_layers=model.config.num_hidden_layers,
        num_heads=model.config.num_attention_heads,
        head_dim=model.config.head_dim,
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        num_samples=args.num_samples,
        sample_seq_len=args.sample_seq_len,
        n_sink=args.n_sink,
    )
    output_path = os.path.join(args.output_path, stats.stats_id())
    stats.save_pretrained(output_path)

# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import fire
import torch
from datasets import load_dataset
from huggingface_hub import PyTorchModelHubMixin, get_collection
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
    dataset_name : str, default="kmfoda/booksum"
        Dataset used to compute the statistics.
    num_samples : int, default=100
        Number of samples used to compute the statistics.
    sample_seq_len : int, default=1000
        Sequence length used to compute the statistics.
    """

    # Override parent defaults to enable stats by default
    sample_seq_len: int = 1000
    num_samples: int = 100
    dataset_name: str = "kmfoda/booksum"
    stats_folder: Optional[str] = None

    mu: torch.Tensor = field(init=False, default=None)  # initialized in __post_init_from_model__
    cov: torch.Tensor = field(init=False, default=None)  # initialized in __post_init_from_model__

    def get_query_statistics(self, module: nn.Module, hidden_states: torch.Tensor):
        """
        Override the parent method to use the pre-computed query statistics.
        """
        q_len = hidden_states.shape[1]
        layer_idx = module.layer_idx
        mu, cov = self.apply_avg_rope(module, self.mu[layer_idx], self.cov[layer_idx], q_len)  # type: ignore
        return mu.unsqueeze(0), cov.unsqueeze(0)

    @staticmethod
    def available_stats():
        collection = get_collection("alessiodevoto/expectedattentionstats-68b0248d519303713320e2cf")
        return [x.item_id for x in collection.items]

    def __post_init_from_model__(self, model):
        """
        Automatically load or compute query statistics for the model.
        """
        if self.mu is None and self.cov is None:
            if self.stats_folder is not None:
                stats = ExpectedAttentionStats.from_pretrained(self.stats_folder)
            else:
                stats = self._maybe_load_stats_from_hub(model)
            self.mu = stats.query_mean.data.to(model.device, dtype=model.dtype)
            self.cov = stats.query_cov.data.to(model.device, dtype=model.dtype)

    def _maybe_load_stats_from_hub(self, model: PreTrainedModel):
        """Load statistics from the Hugging Face Hub."""
        stats_id = ExpectedAttentionStats(
            model_name=model.config.name_or_path,
            num_layers=model.config.num_hidden_layers,
            num_heads=model.config.num_attention_heads,
            head_dim=model.config.head_dim,
            dataset_name=self.dataset_name,
            num_samples=self.num_samples,
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
        captured_tensors.append(q_embed.detach().cpu())
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
    sample_seq_len: int,
    n_sink: int,
    text_column: str = "chapter",
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
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
        text_column (str): Name of the column in the dataset containing the text to tokenize.

    Returns:
        list or tuple:
            collected_queries (list): List of query tensors, each of shape (num_layers, num_heads, seq_len, head_dim)
            mean_query (torch.Tensor): Mean query vector for each layer and head.
            cov_query (torch.Tensor): Covariance matrix of queries for each layer and head.
    """

    # Load dataset and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
    dataset = load_dataset(dataset_name, split=f"train[:{num_samples}]")

    # Cut to max q_len
    dataset = dataset.map(lambda x: {text_column: x[text_column][:sample_seq_len]})

    collected_queries = []
    for text in tqdm(dataset[text_column], desc="Collecting queries"):
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with patch_rotary_embedding(model) as captured_queries:
            model(**inputs)
        collected_queries.append(torch.cat(captured_queries, dim=0)[:, :, n_sink:, :])

    cat_queries = torch.cat(collected_queries, dim=-2)
    mean_query = cat_queries.mean(dim=-2)
    # compute covariance manually
    centered_queries = cat_queries - mean_query.unsqueeze(-2)
    N = cat_queries.shape[-2]
    cov_query = (centered_queries.transpose(-2, -1) @ centered_queries) / (N - 1)
    return collected_queries, mean_query, cov_query


def main(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    output_path: str = ".",
    dataset_name: str = "kmfoda/booksum",
    num_samples: int = 100,
    sample_seq_len: int = 1000,
    n_sink: int = 4,
    text_column: str = "chapter",
    device_map: str = "auto",
):
    """
    Collect query statistics for a transformer model and save them.

    Args:
        model_name: Name of the model to collect statistics for
        output_path: Directory to save the statistics
        dataset_name: Dataset to use for collecting statistics
        num_samples: Number of samples to use from the dataset
        sample_seq_len: Sequence length for each sample
        n_sink: Number of initial tokens to exclude
        text_column: Column name containing text in the dataset
        device_map: Device mapping for the model
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, dtype=torch.bfloat16).eval()

    _, mu, cov = collect_queries(model, dataset_name, num_samples, sample_seq_len, n_sink, text_column)

    stats = ExpectedAttentionStats(
        num_layers=model.config.num_hidden_layers,
        num_heads=model.config.num_attention_heads,
        head_dim=model.config.head_dim,
        dataset_name=dataset_name,
        model_name=model_name,
        num_samples=num_samples,
        sample_seq_len=sample_seq_len,
        n_sink=n_sink,
    )
    stats.query_mean.data = mu
    stats.query_cov.data = cov

    output_path = os.path.join(output_path, stats.stats_id())
    stats.save_pretrained(output_path)
    print(f"Statistics saved to: {output_path}")


if __name__ == "__main__":
    fire.Fire(main)

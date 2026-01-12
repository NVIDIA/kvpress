# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Training script for KVzap models.

This module provides functions to train KVzap models (MLP and Linear) that predict
KVzip+ importance scores from hidden states. The trained models can be used with
KVzapPress to compress the KV cache during inference.
"""

import numpy as np
from pathlib import Path

from tqdm.auto import tqdm

import torch
from torch import nn

from skorch import NeuralNetRegressor
from skorch.callbacks import LRScheduler, GradientNormClipping
from skorch.dataset import ValidSplit
from sklearn.linear_model import Ridge

from transformers import AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config

from kvpress.presses.kvzap_press import KVzapModel, KVzapConfig
from kvzap.data import load_nemotron_dataset, KVzapDataCollector


def train_mlp(
    X: torch.Tensor,
    y: torch.Tensor,
    hidden_dim: int,
    device: str,
    max_epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 512,
) -> KVzapModel:
    """
    Train a two-layer MLP model to predict KVzip+ scores from hidden states.

    Parameters
    ----------
    X : torch.Tensor
        Input hidden states of shape (n_samples, n_layers, hidden_size)
    y : torch.Tensor
        Target scores of shape (n_samples, n_layers, n_kv_heads)
    hidden_dim : int
        Hidden dimension of the MLP
    device : str
        Device to train on (e.g., "cuda:0")
    max_epochs : int, optional
        Maximum training epochs, by default 10
    lr : float, optional
        Learning rate, by default 1e-3
    batch_size : int, optional
        Batch size, by default 512

    Returns
    -------
    KVzapModel
        Trained MLP model
    """
    mlp = KVzapModel(
        KVzapConfig(input_dim=X.shape[2], hidden_dim=hidden_dim, output_dim=y.shape[2], n_modules=X.shape[1])
    )
    mlp.to(device, dtype=X.dtype)

    net = NeuralNetRegressor(
        mlp,
        max_epochs=max_epochs,
        criterion=nn.MSELoss(),
        lr=lr,
        optimizer=torch.optim.AdamW,
        iterator_train__shuffle=True,
        device=device,
        batch_size=batch_size,
        callbacks=[
            LRScheduler(policy="CosineAnnealingLR", T_max=max_epochs),
            GradientNormClipping(gradient_clip_value=1.0),
        ],
        train_split=ValidSplit(0.05, random_state=42),
    )

    net.fit(X, y)
    return mlp


def train_linear(X: torch.Tensor, y: torch.Tensor) -> KVzapModel:
    """
    Train a linear model to predict KVzip+ scores from hidden states.

    Parameters
    ----------
    X : torch.Tensor
        Input hidden states of shape (n_samples, n_layers, hidden_size)
    y : torch.Tensor
        Target scores of shape (n_samples, n_layers, n_kv_heads)

    Returns
    -------
    KVzapModel
        Trained linear model
    """
    # Train a linear model for each layer
    params = []
    for layer_idx in tqdm(range(X.shape[1]), desc="Training linear models"):
        linear = Ridge()
        linear.fit(X[:, layer_idx].float(), y[:, layer_idx].float())
        params.append((linear.coef_, linear.intercept_))

    # Load the parameters into a KVzapModel
    linear_model = KVzapModel(
        KVzapConfig(input_dim=X.shape[2], hidden_dim=None, output_dim=y.shape[2], n_modules=X.shape[1])
    )
    for layer_idx, (W, b) in enumerate(params):
        linear_model.layers[layer_idx].weight.data = torch.tensor(W, dtype=X.dtype)  # type: ignore[index]
        linear_model.layers[layer_idx].bias.data = torch.tensor(b, dtype=X.dtype)  # type: ignore[index]
    return linear_model


def train(
    model_name: str,
    output_dir: str,
    # Dataset parameters
    min_tokens: int = 750,
    max_tokens: int = 1250,
    n_train_per_subset: int = 500,
    n_test_per_subset: int = 5,
    n_tokens: int = 500,
    fp8: bool = False,
    # MLP training parameters
    hidden_dim: int = 512,
    max_epochs: int = 15,
    lr: float = 5e-3,
    batch_size: int = 512,
    device: str = "cuda:0",
):
    """
    Train KVzap models (MLP and linear) for a given language model.

    This function:
    1. Loads the model and tokenizer
    2. Loads and preprocesses the Nemotron dataset
    3. Extracts KVzip+ scores using the repeat prompt method
    4. Trains both 2-layer MLP and linear models
    5. Saves models and predictions to the output directory

    Parameters
    ----------
    model_name : str
        HuggingFace model name (e.g., "Qwen/Qwen3-8B")
    output_dir : str
        Directory to save trained models and predictions
    min_tokens : int, optional
        Minimum tokens per sample, by default 750
    max_tokens : int, optional
        Maximum tokens per sample, by default 1250
    n_train_per_subset : int, optional
        Training samples per dataset subset, by default 500
    n_test_per_subset : int, optional
        Test samples per dataset subset, by default 5
    n_tokens : int, optional
        Tokens to sample per text sample, by default 500
    fp8 : bool, optional
        Whether to use FP8 quantization to run the model, by default False
    hidden_dim : int, optional
        Hidden dimension for MLP model, by default 512
    max_epochs : int, optional
        Maximum training epochs for MLP, by default 15
    lr : float, optional
        Learning rate for MLP training, by default 5e-3
    batch_size : int, optional
        Batch size for MLP training, by default 512
    device : str, optional
        Device to use for training the MLP, by default "cuda:0"
    """
    # Verify input parameters
    assert n_tokens < min_tokens, "n_tokens must be less than min_tokens"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    assert output_path.is_dir() and not list(output_path.iterdir()), "Output directory is not empty"

    # Load model and tokenizer
    print(f"Loading model {model_name} and tokenizer")
    quantization_config = FineGrainedFP8Config() if fp8 else None
    model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype="auto",
            device_map="auto",
            attn_implementation="eager",
            quantization_config=quantization_config,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load dataset
    print("Loading dataset")
    df = load_nemotron_dataset(tokenizer, min_tokens, max_tokens, n_train_per_subset, n_test_per_subset)
    print(f"Loaded {len(df)} samples (train: {(df['split'] == 'train').sum()}, test: {(df['split'] == 'test').sum()})")

    # Extract scores using KVzapDataCollector
    print("Extracting KVzip+ scores")
    collector = KVzapDataCollector(model, tokenizer)
    X, y = collector.collect(df, n_tokens)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    # Split data into train and test
    n_test = n_tokens * (df["split"] == "test").sum()
    X_train, X_test = X[n_test:], X[:n_test]
    y_train, y_test = y[n_test:], y[:n_test]

    # Train MLP and linear models
    print("Training MLP and linear models")
    mlp = train_mlp(X_train, y_train, hidden_dim, device, max_epochs, lr, batch_size)
    linear = train_linear(X_train, y_train)
    linear.to(device)

    # Evaluate and save models and predictions
    print("Evaluating and saving models and predictions")
    for module, name in [(mlp, "mlp"), (linear, "linear")]:
        with torch.no_grad():
            y_pred = module(X_test.to(device))
        # Save model and predictions
        module.save_pretrained(output_path / name)
        np.save(output_path / name / "true.npy", y_test.cpu().float().numpy())
        np.save(output_path / name / "pred.npy", y_pred.cpu().float().numpy())

    print(f"Training complete. Models saved to {output_path}")


if __name__ == "__main__":
    import fire

    fire.Fire(train)

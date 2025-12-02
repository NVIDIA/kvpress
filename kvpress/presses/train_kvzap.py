# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
from pathlib import Path

from tqdm.auto import tqdm

import torch
from torch import nn

try:
    from skorch import NeuralNetRegressor
    from skorch.callbacks import LRScheduler, GradientNormClipping
    from skorch.dataset import ValidSplit

    from sklearn.linear_model import Ridge
except ImportError:
    raise ImportError("skorch or scikit-learn is not installed. Please install them with `pip install skorch`")

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import repeat_kv
from datasets import load_dataset

from kvpress.presses.kvzap_press import KVzapModel, KVzapConfig


def load_nemotron_dataset(tokenizer, min_tokens, max_tokens, n_train_per_subset, n_test_per_subset):
    """
    1. Load the nvidia/Nemotron-Pretraining-Dataset-sample dataset
    2. Keep only samples with a sequence length in [min_tokens, max_tokens]
    3. Split in train/test and keep at most n_[train/test]_per_subset samples per subset for [train/test]

    Motivation for 1: this dataset is both multilingual and multi-domain (General Q&A, Math, Code)
    Motivation for 2: to have samples with ~ uniform sequence length so that the attention
    weight denominator during KVzip+ score extraction is not influenced by the sequence length
    Motivation for 3: to have a more balanced dataset across subsets.
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
    for subset in tqdm(subsets):
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


def repeat_prompt_tokenization(tokenizer, prompt):
    """
    Given a prompt, build an extended prompt following the KVzip method:
    ```user: <prompt>

    Repeat the previous context exactly.
    assistant: <prompt>```

    Returns both the tokenized input_ids and the start and end indexes of the prompt and the repeated prompt
    """

    # Repeat the prompt using the chat template
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
    start_prompt = torch.where(m >= len(prefix))[0][0].item()
    end_prompt = torch.where(m >= len(prefix) + len(prompt))[0][0].item()
    start_repeated_prompt = torch.where(m >= len(prefix) + len(prompt) + len(repeat))[0][0].item()
    end_repeated_prompt = torch.where(m >= len(prefix) + 2 * len(prompt) + len(repeat))[0][0].item()

    # Make sure the indexes are correct
    first_prompt = tokenizer.decode(outputs.input_ids[0][start_prompt:end_prompt])
    repeated_prompt = tokenizer.decode(outputs.input_ids[0][start_repeated_prompt:end_repeated_prompt])
    # assert first_prompt.strip() == prompt.strip() and repeated_prompt.strip() == prompt.strip()

    return outputs.input_ids, start_prompt, end_prompt, start_repeated_prompt, end_repeated_prompt


def forward_hook(module, input, kwargs, output):
    """
    Forward hook to extract the KVzip+ scores from the extended prompt
    Results are stored in the global variable DATA as a list of tuples (hidden_states, scores)
    """

    global START_PROMPT, END_PROMPT, START_REPEATED_PROMPT, END_REPEATED_PROMPT, DATA

    # Get variables
    hidden_states = kwargs["hidden_states"]
    values = kwargs["past_key_values"].layers[module.layer_idx].values
    attn_weights = output[1]

    # Iinialize scores with attention weights
    scores = attn_weights

    # Divide by ||h|| (by row)
    h_norm = torch.norm(hidden_states, dim=-1)
    scores = torch.einsum("b h t i, b t -> b h t i", scores, 1 / h_norm)

    # Multiply by ||WoV|| (by column)
    Wo = module.o_proj.weight.transpose(0, 1)
    Wo = Wo.view(module.config.num_attention_heads, module.head_dim, module.config.hidden_size)
    V = repeat_kv(values, module.num_key_value_groups)
    WoV_norm = torch.einsum("h i j, b h t i -> b h t j", Wo, V).norm(dim=-1)
    scores = torch.einsum("b h t i, b h i -> b h t i", scores, WoV_norm)

    # Get max for each prompt across the repeated prompt tokens and the KV groups
    scores = scores[:, :, START_REPEATED_PROMPT:END_REPEATED_PROMPT, START_PROMPT:END_PROMPT].amax(dim=2)
    scores = scores.view(
        scores.shape[0], module.config.num_key_value_heads, module.num_key_value_groups, scores.shape[2]
    ).amax(dim=2)

    # Apply log
    scores = torch.log(scores)

    # Store the results in the global variable DATA
    DATA.append((hidden_states[0, START_PROMPT:END_PROMPT, :].cpu(), scores[0].T.cpu()))

    return output


def extract_kvzip_scores(model, tokenizer, df, n_tokens):
    """
    Extract pairs (X=hidden_states, y=scores) by running the model on the text samples in the dataset
    For each text sample, randomly sample n_tokens tokens
    """
    global DATA, START_PROMPT, END_PROMPT, START_REPEATED_PROMPT, END_REPEATED_PROMPT

    n_layers = model.model.config.num_hidden_layers
    X = torch.zeros(len(df) * n_tokens, n_layers, model.model.config.hidden_size, dtype=model.dtype)
    y = torch.zeros(len(df) * n_tokens, n_layers, model.model.config.num_key_value_heads, dtype=model.dtype)

    for i, text in tqdm(enumerate(df["text"]), total=len(df)):

        # Extract a text of n_tokens tokens from the sample
        tokens = tokenizer(text)["input_ids"]

        # Get the scores
        tokens, START_PROMPT, END_PROMPT, START_REPEATED_PROMPT, END_REPEATED_PROMPT = repeat_prompt_tokenization(
            tokenizer, text
        )
        DATA = []
        with torch.no_grad():
            model.model(tokens.to(model.device))

        # Sample n_tokens tokens
        mask = torch.randperm(len(DATA[0][0]))[:n_tokens]
        for layer_idx, (X_, y_) in enumerate(DATA):
            X[i * n_tokens : (i + 1) * n_tokens, layer_idx] = X_[mask]
            y[i * n_tokens : (i + 1) * n_tokens, layer_idx] = y_[mask]

    return X, y


def train_mlp(X, y, hidden_dim, device, max_epochs=10, lr=1e-3, batch_size=512):
    """
    Train a 1-layer MLP model, all layers at once
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


def train_linear(X, y):
    """
    Train a linear model, layer by layer
    """

    # Train a linear model for each layer
    params = []
    for layer_idx in tqdm(range(X.shape[1])):
        linear = Ridge()
        linear.fit(X[:, layer_idx].float(), y[:, layer_idx].float())
        params.append((linear.coef_, linear.intercept_))

    # Load the parameters into a KVzapModel
    linear = KVzapModel(KVzapConfig(input_dim=X.shape[2], hidden_dim=None, output_dim=y.shape[2], n_modules=X.shape[1]))
    for layer_idx, (W, b) in enumerate(params):
        linear.module_list[layer_idx].weight.data = torch.tensor(W, dtype=X.dtype)
        linear.module_list[layer_idx].bias.data = torch.tensor(b, dtype=X.dtype)
    return linear


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")

    # Dataset parameters
    parser.add_argument("--min_tokens", type=int, default=750)
    parser.add_argument("--max_tokens", type=int, default=1250)
    parser.add_argument("--n_train_per_subset", type=int, default=500)
    parser.add_argument("--n_test_per_subset", type=int, default=5)
    parser.add_argument("--n_tokens", type=int, default=500)

    # MLP KVzap training parameters
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--max_epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    # Verify input parameters
    assert args.n_tokens < args.min_tokens, "n_tokens must be less than min_tokens"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assert output_dir.is_dir() and not list(output_dir.iterdir()), "Output directory is not empty"

    # Load model and tokenizer
    print(f"Loading model {args.model_name} and tokenizer")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16, attn_implementation="eager")
    model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    for layer in model.model.layers:
        layer.self_attn.register_forward_hook(forward_hook, with_kwargs=True)

    # Load dataset
    print(f"Loading dataset")
    df = load_nemotron_dataset(
        tokenizer, args.min_tokens, args.max_tokens, args.n_train_per_subset, args.n_test_per_subset
    )
    print(f"Loaded {len(df)} samples (train: {(df['split'] == 'train').sum()}, test: {(df['split'] == 'test').sum()})")

    # Extract scores
    print(f"Extracting KVzip+ scores")
    X, y = extract_kvzip_scores(model, tokenizer, df, args.n_tokens)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    # Split data into train and test
    n_test = args.n_tokens * (df["split"] == "test").sum()
    X_train, X_test = X[n_test:], X[:n_test]
    y_train, y_test = y[n_test:], y[:n_test]

    # Train MLP and linear models
    print(f"Training MLP and linear models")
    mlp = train_mlp(X_train, y_train, args.hidden_dim, args.device, args.max_epochs, args.lr, args.batch_size)
    linear = train_linear(X_train, y_train)
    linear.to(args.device)

    # Evaluate and save models and predictions
    print(f"Evaluating and saving models and predictions")
    for module, name in [(mlp, "mlp"), (linear, "linear")]:
        with torch.no_grad():
            y_pred = module(X_test.to(args.device))
        # Save model and predictions
        module.save_pretrained(output_dir / name)
        np.save(output_dir / name / "true.npy", y_test.cpu().float().numpy())
        np.save(output_dir / name / "pred.npy", y_pred.cpu().float().numpy())

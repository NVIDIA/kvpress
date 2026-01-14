# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import uuid
from tqdm import tqdm
from pathlib import Path
from contextlib import nullcontext

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from kvpress import KVzapPress, ThresholdPress


def calculate_metrics(df):
    """
    Calculate metrics for the AIME25 benchmark.
    """
    correct = 0
    answered = 0
    for _, row in df.iterrows():
        try:
            y_pred = str(row["predicted_answer"].split("boxed{")[-1].split("}")[0])
            y_true = str(row["answer"])
            score = int(y_pred == y_true)
        except IndexError:
            score = 0
        correct += score
        answered += "boxed{" in row["predicted_answer"]
    return {"correct": correct, "answered": answered, "accuracy": correct / len(df), "total": len(df)}


def evaluate(
    kvzap_model_type: str,
    threshold: float = 0.0,
    model_name: str = "Qwen/Qwen3-8B",
    device: str = "cuda:0",
    max_new_tokens: int = 32000,
):
    """Evaluate KVzap on the AIME25 benchmark using model.generate instead of the
    KVpress pipeline in order to use sampling parameters and not greedy decoding.

    Parameters
    ----------
    kvzap_model_type : str
        Model type - "mlp", "linear", or "no_press"
    threshold : float, optional
        Threshold for KVzap scores, by default 0.0
    model_name : str, optional
        HuggingFace model name, by default "Qwen/Qwen3-8B"
    device : str, optional
        Device to use, by default "cuda:0"
    max_new_tokens : int, optional
        Maximum number of tokens to generate, by default 32000
    """

    # Create press
    press: ThresholdPress | type[nullcontext[None]]
    if kvzap_model_type == "no_press":
        press = nullcontext
    else:
        press = ThresholdPress(
            KVzapPress(model_type=kvzap_model_type),
            threshold=threshold,
            decoding=True,
        )

    # Load tokenizer, model and dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto").to(device)
    df = load_dataset("alessiodevoto/aime25", split="test").to_pandas()

    # Run evaluation
    for idx, row in tqdm(df.iterrows(), total=len(df)):

        # Tokenize question
        messages = [{"role": "user", "content": row["question"]}]
        tokens = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        tokens = tokens.to(model.device)

        with press(model):
            # Generation config from model card: https://huggingface.co/Qwen/Qwen3-32B
            output_tokens = model.generate(
                tokens, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, max_new_tokens=max_new_tokens
            )
            answer = tokenizer.decode(output_tokens[0, tokens.shape[1] :])
            df.loc[idx, "predicted_answer"] = answer
            if isinstance(press, ThresholdPress):
                df.loc[idx, "compression_ratio"] = press.compression_ratio
            else:
                df.loc[idx, "compression_ratio"] = 0

    # Save results in a new directory
    dir_id = uuid.uuid4().hex
    output_dir = Path(
        f"results/aime25__{model_name.replace('/', '--')}__kvzap_{kvzap_model_type}__{threshold:.2f}/{dir_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "predictions.csv", index=False)

    # Calculate and save metrics
    metrics = calculate_metrics(df)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)

    print(f"Results saved to {output_dir}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    import fire

    fire.Fire(evaluate)

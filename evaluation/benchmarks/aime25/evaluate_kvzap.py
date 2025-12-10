import uuid
from tqdm import tqdm
from pathlib import Path
from contextlib import nullcontext
from typing import Any, ContextManager
from collections.abc import Callable

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from kvpress import KVzapPress, ThresholdPress


def evaluate(press, model_name, device, max_new_tokens):  # type: ignore[type-arg]
    """
    Evaluate the press without the kvpress pipeline but the model.generate method.
    This allows to avoid greedy decoding which is not recommended for reasoning
    """

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
            if press == nullcontext:
                df.loc[idx, "compression_ratio"] = 0
            else:
                df.loc[idx, "compression_ratio"] = press.compression_ratio
    return df


if __name__ == "__main__":

    import argparse
    import json
    from calculate_metrics import calculate_metrics

    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--model_type", type=str, choices=["mlp", "linear", "no_press"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B", choices=["Qwen/Qwen3-8B", "Qwen/Qwen3-32B"])
    parser.add_argument("--max_new_tokens", type=int, default=32000)

    args = parser.parse_args()

    press: Callable[..., ContextManager[Any]]
    if args.model_type == "no_press":
        press = nullcontext
    else:
        press = ThresholdPress(
            KVzapPress(model_type=args.model_type),
            threshold=args.threshold,
            decoding=True,
        )

    # Run evaluation
    df = evaluate(press, args.model_name, args.device, args.max_new_tokens)

    # Save results
    dir_id = uuid.uuid4().hex
    output_dir = Path(
        f"results/aime25__{args.model_name.replace('/', '--')}__kvzap_{args.model_type}__{args.threshold:.2f}/{dir_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "predictions.csv", index=False)

    # Calculate metrics
    metrics = calculate_metrics(df)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)

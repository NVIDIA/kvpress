# KVzap

[![KVzap collection](https://img.shields.io/badge/🤗%20Hugging%20Face-Collection-orange)](https://huggingface.co/collections/nvidia/kvzap) 
[![arXiv](https://img.shields.io/badge/arXiv-2601.07891-b31b1b.svg)](https://arxiv.org/abs/2601.07891)

[KVzap](https://arxiv.org/abs/2601.07891) is a fast approximation of [KVzip](https://arxiv.org/abs/2505.23416) that works in both prefilling and decoding. It applies a lightweight surrogate model to the hidden states to predict importance scores, and removes the KV pairs with a score below a given threshold.

## Usage

KVzap is designed to be used by combining the `KVzapPress` and the `ThresholdPress` from kvpress:

```python
import requests
from transformers import pipeline
from kvpress import KVzapPress, ThresholdPress

model = "Qwen/Qwen3-8B"
pipe = pipeline("kv-press-text-generation", model=model, device_map="auto", dtype="auto")
press = ThresholdPress(KVzapPress(model_type="mlp"), threshold=-4)

# Prefilling compression only, thinking disabled
press.decoding = False
context = requests.get("https://arxiv.org/abs/2601.07891").text
question = "\n What is this article about in 2 sentences ?"
answer = pipe(context, question=question, press=press)["answer"]
print(f"Compression ratio: {press.compression_ratio:.2%}\nAnswer: {answer}")

# Prefilling and decoding compression, thinking enabled
press.decoding = True
prompt = "What is the best hardware to run LLMs and why ?"
answer = pipe(prompt, press=press, enable_thinking=True, max_new_tokens=2000)["answer"]
print(f"Compression ratio: {press.compression_ratio:.2%}\nAnswer: {answer}")
```

The `KVzapPress` inherits from the `ScorerPress` class and only predicts the scores for every KV pair. The `ThresholdPress` then prunes the KV pairs with a score below a given threshold, rather than using a fixed compression ratio.

Supported base models are provided in the [KVzap collection](https://huggingface.co/collections/nvidia/kvzap) but can easily be extended to any other model following the instructions in the [training section](#training).

## Training

Training uses the [Nemotron-Pretraining-Dataset-sample](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample) to extract KVzip+ scores and train surrogate models. 

To reproduce the training or train your own model, use the following command:

```bash
pip install skorch scikit-learn
python train.py --model_name <model_name> --output_dir <output_dir>
```

Run `python train.py --help` for all options.

## Evaluation

Evaluation can be reproduced by using the [kvpress evaluation CLI](../evaluation). 

We provide a specific script to evaluate KVzap on the AIME25 benchmark using `model.generate` directly to enable sampling-based decoding rather than greedy decoding:

```bash
python evaluate_aime.py <model_type> --threshold <threshold> --model_name <base_model_name>
```

where `<model_type>` is the type of KVzap model to use ("mlp", "linear" or "no_press") and `<base_model_name>` the name of the base model to use (e.g. "Qwen/Qwen3-8B").

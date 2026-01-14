# KVzap

[![KVzap collection](https://img.shields.io/badge/🤗%20Hugging%20Face-Collection-orange)](https://huggingface.co/collections/nvidia/kvzap) 
[![arXiv](https://img.shields.io/badge/arXiv-2601.07891-b31b1b.svg)](https://arxiv.org/abs/2601.07891)

[KVzap](https://arxiv.org/abs/2601.07891) is a fast approximation of [KVzip](https://arxiv.org/abs/2505.23416) that works in both prefilling and decoding. It applies a lightweight surrogate model to the hidden states to predict importance scores, and removes the KV pairs with a score below a given threshold.

## Usage

KVzap is designed to be used by combining the `KVzapPress` and the `ThresholdPress` from kvpress:

```python
from kvpress import KVzapPress, ThresholdPress

press = ThresholdPress(KVzapPress(model_type="mlp"), threshold=-4, decoding=True)

with press(model):
    outputs = model.generate(inputs)

print(press.compression_ratio)
```

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

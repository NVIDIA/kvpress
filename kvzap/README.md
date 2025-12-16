# KVzap

[![KVzap collection](https://img.shields.io/badge/🤗%20Hugging%20Face-Collection-orange)](https://huggingface.co/collections/nvidia/kvzap)

KVzap approximates KVzip+ (an improved version of [KVzip](https://arxiv.org/abs/2505.23416)) by training a lightweight surrogate model on top of the hidden states.

## Training

Training uses the [Nemotron-Pretraining-Dataset-sample](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample) to extract KVzip+ scores and train surrogate models. Pre-trained models are available in the [KVzap collection](https://huggingface.co/collections/nvidia/kvzap).

To reproduce the training, you can use the following command:

```bash
pip install skorch scikit-learn
python train.py --model_name <model_name> --output_dir <output_dir>
```

Run `python train.py --help` for all options.

## Evaluation on AIME25

This script evaluates KVzap on the AIME25 benchmark using `model.generate` directly - instead of the kvpress pipeline - to enable sampling-based decoding rather than greedy decoding.

```bash
python evaluate_aime.py mlp --threshold -4 --model_name Qwen/Qwen3-8B
```

## Usage

KVzap is designed to be used in conjunction with the `ThresholdPress` from kvpress. You can use the following code to compress the KV cache during inference:

```python
from kvpress import KVzapPress, ThresholdPress

press = ThresholdPress(KVzapPress(model_type="mlp"), threshold=-4, decoding=True)

with press(model):
    output = model.generate(input_ids)
```

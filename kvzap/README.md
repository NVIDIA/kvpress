# KVzap

[![KVzap collection](https://img.shields.io/badge/🤗%20Hugging%20Face-Collection-orange)](https://huggingface.co/collections/nvidia/kvzap)

KVzap approximates KVzip+ (an improved version of [KVzip](https://arxiv.org/abs/2505.23416)) by training a lightweight auxiliary model on top of the hidden states.

## Directory Contents

- `train.py`: Training script for the KVzap auxiliary model (Linear or MLP)
- `evaluate_aime.py`: Evaluation script for KVzap on the AIME25 benchmark

## Training

The training script extracts KVzip+ scores from text samples and trains either a linear or MLP model to predict these scores from hidden states.

### Requirements

Install additional dependencies:

```bash
pip install skorch scikit-learn
```

### Usage

```bash
python train.py \
    --model_name <model_name> \
    --output_dir <output_dir> \
    --device cuda:0
```

#### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_name` | HuggingFace model name | (required) |
| `--output_dir` | Directory to save trained models | (required) |
| `--device` | Device to use | `cuda:0` |
| `--min_tokens` | Minimum tokens per sample | `750` |
| `--max_tokens` | Maximum tokens per sample | `1250` |
| `--n_train_per_subset` | Training samples per subset | `500` |
| `--n_test_per_subset` | Test samples per subset | `5` |
| `--n_tokens` | Tokens to sample per text | `500` |
| `--hidden_dim` | MLP hidden dimension | `512` |
| `--max_epochs` | Training epochs | `15` |
| `--lr` | Learning rate | `5e-3` |
| `--batch_size` | Batch size | `512` |

## Evaluation on AIME25

The evaluation script tests KVzap compression on the AIME25 mathematical reasoning benchmark.

### Usage

```bash
python evaluate_aime.py mlp --threshold 0.0 --model_name Qwen/Qwen3-8B --device cuda:0
```

#### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `kvzap_model_type` | Model type: `mlp`, `linear`, or `no_press` | (required) |
| `--threshold` | Threshold for KVzap scores | `0.0` |
| `--model_name` | Model to evaluate | `Qwen/Qwen3-8B` |
| `--device` | Device to use | `cuda:0` |
| `--max_new_tokens` | Maximum generation tokens | `32000` |

## Using KVzapPress

```python
from kvpress import KVzapPress, ThresholdPress

# Create press with MLP model
press = ThresholdPress(
    KVzapPress(model_type="mlp"),
    threshold=-3,
    decoding=True,
)

# Use with model
with press(model):
    output = model.generate(input_ids)
```


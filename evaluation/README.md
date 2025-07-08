# Evaluation

We support simple evaluation for all the presses implemented in the library on a variety of popular benchmarks.

## Quick Start

Running evaluation is straightforward:

1. **Configure your evaluation** - Edit `evaluate_config.yaml` to specify your *method*, *press*, and *dataset*
2. **Run the evaluation** - Execute the script: ```python evaluate.py```

The script will read from `evaluate_config.yaml` and run inference accordingly. 
If you want, you can override the settings defined in `evaluate_config.yaml` via command line: For instance:

```bash
python evaluate.py --dataset loogle --data_dir shortdep_qa --model meta-llama/Meta-Llama-3.1-8B-Instruct --press_name expected_attention --compression_ratio 0.5
```
> 💡 Results (predictions & metrics) are automatically saved to the `output_dir` directory .


### Configuration

Customize your evaluation by editing `evaluate_config.yaml`. This allows you to configure a number of settings, like the fraction of dataset to use (for quick testing) and the model arguments.
For complete parameter details, see the `evaluation_config.yaml`


### Available Presses and Datasets
We support evaluation with all the presses implemented in the library (and possible combinations). 

- All implemented presses are listed in the `PRESS_REGISTRY` variable in `evaluate_registry.py`.
- All implemented dataset are listed in `DATASET_REGISTRY` variable in `evaluate_registry.py`. 

At the moment, we support the following standard popular benchmarks:

- [Loogle](loogle/README.md) ([hf link](https://huggingface.co/datasets/simonjegou/loogle))
- [RULER](ruler/README.md) ([hf link](https://huggingface.co/datasets/simonjegou/ruler))
- [Zero Scrolls](zero_scrolls/README.md) ([hf link](https://huggingface.co/datasets/simonjegou/zero_scrolls))
- [Infinitebench](infinite_bench/README.md) ([hf link](https://huggingface.co/datasets/MaxJeblick/InfiniteBench))
- [longbench](longbench/README.md)([hf link](https://huggingface.co/datasets/Xnhyacinth/LongBench))
- [longbench-v2](longbenchv2/README.md)([hf link](https://huggingface.co/datasets/Xnhyacinth/LongBench-v2))

📚 **For detailed information** about each dataset or implementing custom benchmarks, see the individual README files in the benchmarks directory.


### Multi GPU Evaluation

Use the provided `evaluate.sh` script to run multiple presses simultaneously across different GPUs with varying compression ratios.
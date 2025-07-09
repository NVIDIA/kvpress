# Evaluation

We support evaluation for all the presses implemented in the library, on a variety of popular benchmarks.

### Quick Start 🚀

Running evaluation is straightforward! Make sure you are in the `evaluation` directory, then:

1. **Configure your evaluation** - Edit `evaluate_config.yaml` to specify your *method*, *press*, and *dataset*
2. **Run the evaluation** - Execute the script: ```python evaluate.py```

The script will read from `evaluate_config.yaml` and run inference accordingly. 
If you want, you can override the settings via command line, for instance:

```bash
python evaluate.py --dataset loogle --data_dir shortdep_qa --model meta-llama/Meta-Llama-3.1-8B-Instruct --press_name expected_attention --compression_ratio 0.5
```

or pass a custom configuration file:

```bash
python evaluate.py --config_path <your_config.yaml>
```

💡 Results (predictions & metrics) are automatically saved to the `output_dir` directory .


### Configuration 

Customize your evaluation by editing `evaluate_config.yaml`. This allows you to flexibly configure a variety of settings, like the `fraction` of dataset to use (for quick testing) and the model arguments (e.g. for scaling RoPE). For complete parameter details, see the `evaluation_config.yaml`


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

### Discussion
The methods benchmarked so far are not able to efficiently compress the KV cache while maintaining performance on several long-context datasets and models.
In particular, exact information retrieval tasks such as kv-retrieval are challenging for the current methods.
Further methods could be explored:
- {Layer,Head}-wise pruning: pruning with a different compression ratio for each layer or head as in [DMC](https://arxiv.org/abs/2403.09636), [FastGen](https://arxiv.org/abs/2310.01801) or [DuoAttention](https://arxiv.org/abs/2410.10819)
- Adaptive pruning: pruning based on a score, and not a uniform fixed ratio
- Taking into account inter-layer dependencies such as in [PyramidKV](https://arxiv.org/abs/2406.02069)
- Move beyond pruning, as this method is fundamentally limited (see last figure in [this notebook](../notebooks/expected_attention.ipynb))
- Fine-tuning LLMs to deal with compressed KV caches

We encourage contributions to explore these ideas and improve the performance of long-context LLMs with compressed caches. We provide benchmark results from 7 presses and 3 models. We include a variant of SnapKV where we include the question in the compression process as in the original paper (snapkv w/ question). All performance curves can be found in the [assets](assets) directory, and predictions are available [here](https://drive.google.com/drive/folders/14BilGw07v8tOUUct-5nDhQlN3zIX9BUf?usp=drive_link).
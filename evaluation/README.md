# Evaluation

This directory contains a set of scripts to evaluate the performance of different presses on different datasets. We currently support the following datasets:
- [Loogle](loogle/README.md) ([hf link](https://huggingface.co/datasets/simonjegou/loogle))
- [RULER](ruler/README.md) ([hf link](https://huggingface.co/datasets/simonjegou/ruler))
- [Zero Scrolls](zero_scrolls/README.md) ([hf link](https://huggingface.co/datasets/simonjegou/zero_scrolls))
- [Infinitebench](infinite_bench/README.md) ([hf link](https://huggingface.co/datasets/MaxJeblick/InfiniteBench))


Please refer to the README of each dataset for more information on how the Hugging Face dataset was generated.

## Usage

To evaluate a press on a dataset, you can run the following command:

```bash
python evaluate.py --dataset <dataset_name> --data_dir <data_dir> --model <model_name> --press_name <press_name> --compression_ratio <ratio> 
```

For instance,
```bash
python evaluate.py --dataset loogle --data_dir shortdep_qa --model meta-llama/Meta-Llama-3.1-8B-Instruct --press_name expected_attention --compression_ratio 0.5
```

- Results (predictions & metrics) are saved in the `results` directory. 
- All available presses are listed in the `PRESS_DICT` variable in `evaluate.py`. 
- Additional arguments are --device, --fraction, --max_new_tokens, --max_context_length and --compress_questions. For more information, run `python evaluate.py --help`
- Finally we also provide a bash script `evaluate.sh` to facilitate the evaluation of multiple presses (1 per GPU) with different compression ratios.


## Benchmarks

We provide benchmark results from 7 presses and 3 models. We include a variant of SnapKV where we include the question in the compression process as in the original paper (snapkv w/ question). All performance curves can be found in the [assets](assets) directory, and predictions are available [here](https://drive.google.com/drive/folders/14BilGw07v8tOUUct-5nDhQlN3zIX9BUf?usp=drive_link).

<details><summary> 

### RULER
</summary>

Average performance the 13 tasks of the RULER dataset with 4k context length (per task results [here](../evaluation/assets/)):

![RULER](../evaluation/assets/ruler_4096_average%20score.png) 

Observations: 
- snapkv w/ question consistently outperforms other methods. However this method can't be use for use cases such as prompt caching as it requires the question to be known beforehand.
- All presses show degradation in performance even for small compression ratios.
- llama3.1-8b-instruct is more robust to compression than other models and expected attention performs better than others.
- mistral-nemo-instruct-2407 is more robust to random pruning than other models.
- For phi-3.5-mini and mistral-nemo-instruct-2407, all presses perform poorly compared to baseline presses such as random (remove KV pairs randomly) or streaming llm (remove the middle KV pairs). This is especially true for the subtask [niah_single_3](assets/ruler_4096_niah_single_3.png) where most presses fail to perform a proper copy-paste of a long needle in a haystack. This might be related to [induction heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- For phi-3.5-mini, we ran an additional experiment with a different compression ratio per layer (as in [this notebook](../notebooks/per_layer_compression_demo.ipynb)) which  largely outperformed it's uniform compression counterpart (see purple cross on 2nd plot). The ratios where determined by grid search on 20/6500 samples from RULER (so results can be questionable).

</details>

<details><summary> 

### Loogle
</summary>

Shortdep_qa
![shortdep_qa](../evaluation/assets/loogle_shortdep_qa.png)
Shortdep_cloze
![shortdep_cloze](../evaluation/assets/loogle_shortdep_cloze.png)
Longdep_qa
![longdep_qa](../evaluation/assets/loogle_longdep_qa.png) 

Observations: 
- Metrics are adapted from loogle benchmark, see [here](../evaluation/loogle/calculate_metrics.py). The plot show the average score (mean over all submetrics) for each task.
- The metrics are not always correlated with the quality of the answer, espcecially for longdep_qa task. LLM-as-a-judge may better suited for a more refined evaluation.
- Again, snapkv w/ question consistently outperforms other methods.
- In longdep_qa, the model looses track on counting (e.g. answer to "How many times is person x mentioned?" gets lower with increased compression ratio). This is not necessarily reflected in the metrics.
- Llama3.1-8b-instruct seems to be more robust to compression.
- Observed attention context had to be truncated at 10 000 tokens to prevent OOM issues, as the attention matrix needs to be materialized.
- For shortdep_cloze task, the output formatting is often ignored leading to performance degradation even for low compression ratios. Interestingly, the model may still be able to answer the question correctly.
- mistral-nemo-instruct-2407 fails to perform well on the shortdep_cloze task, even without compression, and is thus not reported.
- shortdep_cloze task runs OOM for phi-3.5-mini at compression ratio 0.0 and is thus missing.

</details>

### Conclusions

The methods benchmarked so far are not able to efficiently compress the KV cache while maintaining performance on several long-context datasets and models. Further methods could be explored:
- {Layer,Head}-wise pruning: pruning with a different compression ratio for each layer or head as in [DMC](https://arxiv.org/abs/2403.09636), [FastGen](https://arxiv.org/abs/2310.01801) or [DuoAttention](https://arxiv.org/abs/2410.10819)
- Adaptive pruning: pruning based on a score, and not a uniform fixed ratio
- Taking into account inter-layer dependencies such as in [PyramidKV](https://arxiv.org/abs/2406.02069)
- Move beyond pruning, as this method is fundamentally limited (see last figure in [this notebook](../notebooks/expected_attention.ipynb))
- Fine-tuning LLMs to deal with compressed KV caches

We encourage contributions to explore these ideas and improve the performance of long-context LLMs with compressed caches.

## How to add a dataset

Each dataset directory is structured as follows:

```bash
$dataset
├── README.md
├── calculate_metrics.py
├── create_huggingface_dataset.py
```

Where:
- `create_huggingface_dataset.py` is a script that generates the Hugging Face dataset from the original dataset. Each dataset is associated with a set of parquet files with the following structure:
  - `context`: ... 
  - `question`: ...
  - `answer_prefix`: ...
  - `answer`:  ...
  - `max_new_tokens`:  ...
- `calculate_metrics.py` is a script that calculates the metrics based on the output of `evaluate.py`
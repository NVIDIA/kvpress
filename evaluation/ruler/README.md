# RULER dataset

[RULER](https://arxiv.org/abs/2404.06654) generates synthetic examples to evaluate long-context language models with configurable sequence length (from 4k tokens to 128k tokens) and task complexity. It contains a set of 13 tasks grouped in 4 categories (needle in the haystack, question answering, multi-hop tracing and aggregation).

## Hugging Face dataset

The Hugging Face dataset for RULER can be found [here](https://huggingface.co/datasets/simonjegou/ruler). To reproduce this dataset,

1. Install the [RULER repository](https://github.com/hsiehjackson/RULER) and download the necessary data files (see 1. Download data in the README)
2. Copy paste the `generate.sh` from this repository to `$RULER/scripts`, set the `DATA_DIR` variable to your desired location of the RULER data files and run the script
3. Run `create_huggingface_dataset.py` with the correct data_dir and repo_id variables

Notes : by default we use `meta-llama/Meta-Llama-3.1-8B` as the tokenizer, while in the original RULER paper, the tokenizer depends on the model used for evaluation. Results may not be directly comparable to the original RULER benchmark. But as our focus is to evaluate the performance of a given model for different compression ratios, we believe this simplification is acceptable.
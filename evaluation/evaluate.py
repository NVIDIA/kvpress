# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from fire import Fire
from infinite_bench.calculate_metrics import calculate_metrics as infinite_bench_scorer
from loogle.calculate_metrics import calculate_metrics as loogle_scorer
from ruler.calculate_metrics import calculate_metrics as ruler_scorer
from tqdm import tqdm
from transformers import pipeline
from zero_scrolls.calculate_metrics import calculate_metrics as zero_scrolls_scorer

from kvpress import (
    CriticalKVPress,
    CriticalAdaKVPress,
    AdaKVPress,
    ExpectedAttentionPress,
    KnormPress,
    ObservedAttentionPress,
    RandomPress,
    SnapKVPress,
    StreamingLLMPress,
    ThinKPress,
    TOVAPress,
    DuoAttentionPress,
)

logger = logging.getLogger(__name__)

DATASET_DICT = {
    "loogle": "simonjegou/loogle",
    "ruler": "simonjegou/ruler",
    "zero_scrolls": "simonjegou/zero_scrolls",
    "infinitebench": "MaxJeblick/InfiniteBench",
}

SCORER_DICT = {
    "loogle": loogle_scorer,
    "ruler": ruler_scorer,
    "zero_scrolls": zero_scrolls_scorer,
    "infinitebench": infinite_bench_scorer,
}

PRESS_DICT = {
    "criti_adasnapkv": CriticalAdaKVPress(SnapKVPress()),
    "criti_ada_expected_attention": CriticalAdaKVPress(ExpectedAttentionPress(use_vnorm=False)),
    "criti_snapkv": CriticalKVPress(SnapKVPress()),
    "criti_expected_attention": CriticalKVPress(ExpectedAttentionPress(use_vnorm=False)),
    "adasnapkv": AdaKVPress(SnapKVPress()),
    "ada_expected_attention": AdaKVPress(ExpectedAttentionPress()),
    "expected_attention": ExpectedAttentionPress(),
    "ada_expected_attention_e2": AdaKVPress(ExpectedAttentionPress(epsilon=1e-2)),
    "knorm": KnormPress(),
    "observed_attention": ObservedAttentionPress(),
    "random": RandomPress(),
    "snapkv": SnapKVPress(),
    "streaming_llm": StreamingLLMPress(),
    "think": ThinKPress(),
    "tova": TOVAPress(),
    "duo_attention": DuoAttentionPress(),
}


def evaluate(
    dataset: str,
    data_dir: Optional[str] = None,
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    device: Optional[str] = None,
    press_name: str = "expected_attention",
    compression_ratio: float = 0.1,
    fraction: float = 1.0,
    max_new_tokens: Optional[int] = None,
    max_context_length: Optional[int] = None,
    compress_questions: bool = False,
):
    """
    Evaluate a model on a dataset using a press and save the results

    Parameters
    ----------
    dataset : str
        Dataset to evaluate
    data_dir : str, optional
        Subdirectory of the dataset to evaluate, by default None
    model : str, optional
        Model to use, by default "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device : str, optional
        Model device, by default cuda:0 if available else cpu. For multi-GPU use "auto"
    press_name : str, optional
        Press to use (see PRESS_DICT), by default "expected_attention"
    compression_ratio : float, optional
        Compression ratio for the press, by default 0.1
    max_new_tokens : int, optional
        Maximum number of new tokens to generate, by default use the default for the task (recommended)
    fraction : float, optional
        Fraction of the dataset to evaluate, by default 1.0
    max_context_length : int, optional
        Maximum number of tokens to use in the context. By default will use the maximum length supported by the model.
    compress_questions : bool, optional
        Whether to compress the questions as well, by default False
    """

    assert dataset in DATASET_DICT, f"No dataset found for {dataset}"
    assert dataset in SCORER_DICT, f"No scorer found for {dataset}"
    data_dir = str(data_dir) if data_dir else None

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    save_dir = Path(__file__).parent / "results"
    save_dir.mkdir(exist_ok=True)
    save_filename = save_dir / (
        "__".join([dataset, data_dir if data_dir else "", model.replace("/", "--"), press_name, str(compression_ratio)])
        + ".csv"
    )
    if save_filename.exists():
        logger.warning(f"Results already exist at {save_filename}")

    # Load dataframe
    df = load_dataset(DATASET_DICT[dataset], data_dir=data_dir, split="test").to_pandas()
    if fraction < 1.0:
        df = df.sample(frac=fraction, random_state=42)
        save_filename = save_filename.with_name(save_filename.stem + f"__fraction{fraction:.2f}" + save_filename.suffix)

    if max_context_length is not None:
        save_filename = save_filename.with_name(
            save_filename.stem + f"__max_context{max_context_length}" + save_filename.suffix
        )

    if compress_questions:
        df["context"] = df["context"] + df["question"]
        df["question"] = ""
        save_filename = save_filename.with_name(save_filename.stem + "__compressed_questions" + save_filename.suffix)

    # Load press
    assert press_name in PRESS_DICT
    press = PRESS_DICT[press_name]

    if isinstance(press, (DuoAttentionPress)):
        press.head_compression_ratio = compression_ratio
    else:
        press.compression_ratio = compression_ratio  # type:ignore[attr-defined]

    # Initialize pipeline with the correct attention implementation
    model_kwargs = {"torch_dtype": "auto"}
    if isinstance(press, ObservedAttentionPress):
        model_kwargs["attn_implementation"] = "eager"
    else:
        try:
            import flash_attn  # noqa: F401

            model_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass

    if device == "auto":
        pipe = pipeline("kv-press-text-generation", model=model, device_map="auto", model_kwargs=model_kwargs)
    else:
        pipe = pipeline("kv-press-text-generation", model=model, device=device, model_kwargs=model_kwargs)

    # Run pipeline on each context
    df["predicted_answer"] = None
    df_context = df.groupby("context")
    assert all(df_context["answer_prefix"].nunique() == 1)

    for context, df_ in tqdm(df_context, total=df["context"].nunique()):
        questions = df_["question"].to_list()
        max_new_tokens_ = max_new_tokens if max_new_tokens is not None else df_["max_new_tokens"].iloc[0]
        answer_prefix = df_["answer_prefix"].iloc[0]
        output = pipe(
            context,
            questions=questions,
            answer_prefix=answer_prefix,
            press=press,
            max_new_tokens=max_new_tokens_,
            max_context_length=max_context_length,
        )
        df.loc[df_.index, "predicted_answer"] = output["answers"]
        df.loc[df_.index, "compression_ratio"] = press.compression_ratio  # type:ignore[attr-defined]
        torch.cuda.empty_cache()

    # Save answers
    df[["predicted_answer", "compression_ratio"]].to_csv(str(save_filename), index=False)

    # Calculate metrics
    scorer = SCORER_DICT[dataset]
    metrics = scorer(df)
    with open(str(save_filename).replace(".csv", ".json"), "w") as f:
        json.dump(metrics, f)
    print(f"Average compression ratio: {df['compression_ratio'].mean():.2f}")
    print(metrics)


if __name__ == "__main__":
    Fire(evaluate)

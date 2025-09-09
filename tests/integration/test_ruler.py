# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datasets
import pytest
import torch
from transformers import DynamicCache, QuantoQuantizedCache
from transformers.utils import is_flash_attn_2_available, is_optimum_quanto_available

from tests.default_presses import default_presses
from tests.fixtures import kv_press_llama3_1_flash_attn_pipeline  # noqa: F401


@pytest.fixture(scope="session")
def df_ruler():
    df = datasets.load_dataset("simonjegou/ruler", "4096")["test"].to_pandas()
    df = df.loc[df["task"] == "niah_multikey_1"].reset_index(drop=True)
    return df


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
@pytest.mark.skipif(not is_flash_attn_2_available(), reason="flash_attn is not installed")
@pytest.mark.parametrize("press_dict", default_presses)
@pytest.mark.parametrize("cache", ["dynamic", "quantized"])
@pytest.mark.parametrize("compression_ratio", [0, 0.1])
def test_ruler_is_correct(
    kv_press_llama3_1_flash_attn_pipeline, df_ruler, press_dict, cache, compression_ratio  # noqa: F811
):
    cls = press_dict["cls"]
    kwargs = press_dict["kwargs"][0]
    press = cls(**kwargs)
    if not hasattr(cls, "compression_ratio"):
        pytest.skip(reason="Press does not support compression_ratio")
    try:
        # set compression ratio to a small value for testing
        # we don't want to max out compression, but rather test if cache compression works
        press.compression_ratio = compression_ratio
    except AttributeError:
        # pytest.skip(reason="Press does not support setting compression_ratio")
        pass

    if cache == "dynamic":
        cache = DynamicCache()
    elif cache == "quantized" and is_optimum_quanto_available():
        cache = QuantoQuantizedCache(config=kv_press_llama3_1_flash_attn_pipeline.model.config, nbits=4)
    elif cache == "quantized" and not is_optimum_quanto_available():
        pytest.skip("Quanto is not installed")
    else:
        raise ValueError(f"Unknown cache type: {cache}")

    idx = 0
    context = df_ruler.iloc[idx]["context"]
    question = df_ruler.iloc[idx]["question"]
    true_answer = df_ruler.iloc[idx]["answer"][0]

    pred_answer = kv_press_llama3_1_flash_attn_pipeline(context, question=question, press=press, cache=cache)["answer"]
    assert true_answer in pred_answer

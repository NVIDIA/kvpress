import datasets
import pytest
import torch
from transformers import QuantizedCacheConfig, QuantoQuantizedCache
from transformers.utils import is_flash_attn_2_available, is_optimum_quanto_available

from kvpress import ExpectedAttentionPress
from tests.fixtures import kv_press_llama3_1_flash_attn_pipeline  # noqa: F401


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
@pytest.mark.skipif(not is_flash_attn_2_available(), reason="flash_attn is not installed")
def test_kv_press_llama3_1_flash_attn_pipeline(kv_press_llama3_1_flash_attn_pipeline):  # noqa: F811
    df = datasets.load_dataset("simonjegou/ruler", "4096")["test"].to_pandas()
    df = df.loc[df["task"] == "niah_single_3"].reset_index(drop=True)
    press = ExpectedAttentionPress(0.3)

    idx = 0
    context = df.iloc[idx]["context"]
    question = df.iloc[idx]["question"]
    true_answer = df.iloc[idx]["answer"][0]

    pred_answer = kv_press_llama3_1_flash_attn_pipeline(context, question=question, press=press)["answer"]
    assert true_answer in pred_answer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
@pytest.mark.skipif(not is_flash_attn_2_available(), reason="flash_attn is not installed")
@pytest.mark.skipif(not is_optimum_quanto_available(), reason="QuantizedCache is not available")
def test_kv_press_llama3_1_flash_attn_pipeline(kv_press_llama3_1_flash_attn_pipeline):  # noqa: F811
    df = datasets.load_dataset("simonjegou/ruler", "4096")["test"].to_pandas()
    df = df.loc[df["task"] == "niah_single_3"].reset_index(drop=True)
    press = ExpectedAttentionPress(0.15)

    idx = 0
    context = df.iloc[idx]["context"]
    question = df.iloc[idx]["question"]
    true_answer = df.iloc[idx]["answer"][0]

    config = QuantizedCacheConfig(nbits=4)
    cache = QuantoQuantizedCache(config)

    pred_answer = kv_press_llama3_1_flash_attn_pipeline(context, question=question, press=press, cache=cache)["answer"]
    assert true_answer in pred_answer

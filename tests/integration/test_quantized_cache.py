import datasets
import pytest
import torch
from transformers import QuantizedCacheConfig, QuantoQuantizedCache
from transformers.utils import is_optimum_quanto_available

from kvpress import ExpectedAttentionPress
from tests.fixtures import kv_press_llama3_1_flash_attn_pipeline  # noqa: F401

gpu_available = torch.cuda.is_available()
try:
    import flash_attn  # noqa: F401

    flash_attn_installed = True
except:  # noqa: E722
    flash_attn_installed = False


@pytest.mark.skipif(not gpu_available, reason="GPU is not available")
@pytest.mark.skipif(not flash_attn_installed, reason="flash_attn is not installed")
def kv_press_llama3_1_flash_attn_pipeline(kv_press_llama3_1_flash_attn_pipeline):  # noqa: F811
    df = datasets.load_dataset("simonjegou/ruler", "4096")["test"].to_pandas()
    df = df.loc[df["task"] == "niah_single_3"].reset_index(drop=True)
    press = ExpectedAttentionPress(0.3)

    idx = 0
    context = df.iloc[idx]["context"]
    question = df.iloc[idx]["question"]
    true_answer = df.iloc[idx]["answer"][0]

    pred_answer = kv_press_llama3_1_flash_attn_pipeline(context, question=question, press=press)["answer"]
    assert true_answer in pred_answer


@pytest.mark.skipif(not gpu_available, reason="GPU is not available")
@pytest.mark.skipif(not flash_attn_installed, reason="flash_attn is not installed")
@pytest.mark.skipif(not is_optimum_quanto_available(), reason="QuantizedCache is not available")
def kv_press_llama3_1_flash_attn_pipeline(kv_press_llama3_1_flash_attn_pipeline):  # noqa: F811
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

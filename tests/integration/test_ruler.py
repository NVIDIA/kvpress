import datasets
import pytest
import torch
from transformers import DynamicCache, QuantizedCacheConfig, QuantoQuantizedCache
from transformers.utils import is_flash_attn_2_available, is_optimum_quanto_available

from kvpress import ExpectedAttentionPress, KnormPress, SnapKVPress, StreamingLLMPress, ThinKPress, TOVAPress
from tests.fixtures import kv_press_llama3_1_flash_attn_pipeline  # noqa: F401


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is not available")
@pytest.mark.skipif(not is_flash_attn_2_available(), reason="flash_attn is not installed")
@pytest.mark.parametrize(
    "cls", [KnormPress, ExpectedAttentionPress, StreamingLLMPress, SnapKVPress, TOVAPress, ThinKPress]
)
@pytest.mark.parametrize("compression_ratio", [0.1, 0.2])
@pytest.mark.parametrize("cache", ["dynamic", "quantized"])
def test_kv_press_llama3_1_flash_attn_pipeline(
    kv_press_llama3_1_flash_attn_pipeline, cls, compression_ratio, cache  # noqa: F811
):
    df = datasets.load_dataset("simonjegou/ruler", "4096")["test"].to_pandas()
    df = df.loc[df["task"] == "niah_single_3"].reset_index(drop=True)
    if cls == ThinKPress:
        press = cls(key_channel_compression_ratio=compression_ratio, window_size=2)
    else:
        press = cls(compression_ratio=compression_ratio)
    if cache == "dynamic":
        cache = DynamicCache()
    elif cache == "quantized" and is_optimum_quanto_available():
        config = QuantizedCacheConfig(nbits=4)
        cache = QuantoQuantizedCache(config)
    elif cache == "quantized" and not is_optimum_quanto_available():
        pytest.skip("Quanto is not installed")
    else:
        raise ValueError(f"Unknown cache type: {cache}")

    idx = 0
    context = df.iloc[idx]["context"]
    question = df.iloc[idx]["question"]
    true_answer = df.iloc[idx]["answer"][0]

    pred_answer = kv_press_llama3_1_flash_attn_pipeline(context, question=question, press=press, cache=cache)["answer"]
    assert true_answer in pred_answer

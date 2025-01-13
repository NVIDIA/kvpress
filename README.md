[![PyPI version](https://badge.fury.io/py/kvpress.svg)](https://badge.fury.io/py/kvpress)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Colab example notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JNvaTKuuAHrl49dYB9-mdEH_y52Ib-NP?usp=drive_link)

![kvpress](kvpress.jpg)

Deploying long-context LLMs is costly due to the linear growth of the key-value (KV) cache in transformer models. For example, handling 1M tokens with Llama 3.1-70B in float16 requires up to 330GB of memory. This repository implements multiple KV cache compression methods and benchmarks using [🤗 transformers](https://huggingface.co/docs/transformers/en/index), aiming to simplify the development of new methods for researchers and developers in this field.

## Installation

```bash
pip install kvpress
```

We recommend using [flash attention](https://github.com/Dao-AILab/flash-attention/) if possible:
```bash
pip install flash-attn --no-build-isolation
```

## Usage

This repository provides a set of "presses" that compress the KV cache. A press is only applied during the pre-filling phase and is associated with a `compression_ratio` attribute that measures the compression of the cache. The easiest way to use a press is through our custom `KVPressTextGenerationPipeline` that is automatically registered as a transformers pipeline with the name "kv-press-text-generation" when kvpress is imported. It handles chat templates and tokenization for you:

```python
from kvpress import ExpectedAttentionPress
from transformers import pipeline

device = "cuda:0"
model= "microsoft/Phi-3.5-mini-instruct"
pipe = pipeline("kv-press-text-generation", model=model, device=device, torch_dtype="auto", model_kwargs={"attn_implementation":"flash_attention_2"})

context = "A very long text you want to compress once and for all"
question = "\nA question about the compressed context" # optional
    
press = ExpectedAttentionPress(compression_ratio=0.4)
answer = pipe(context, question=question, press=press)["answer"]
```

In the snippet above, the compression is only applied on the context tokens so that you can evaluate the compression for different questions. Check the [Wikipedia notebook demo](notebooks/wikipedia_demo.ipynb) for a more detailed example.

> [!IMPORTANT]  
> We focus on compression during the pre-filling phase as the KV cache becomes a bottleneck for long-context sequence (100k - 1M tokens) which are essentially long context prompts. This would typically apply to improving prompt caching systems.

> [!NOTE]  
> To use the `ObservedAttentionPress`, use `model_kwargs={"attn_implementation":"eager"}` in order to materialize the attention weights (this method is not compatible with flash attention).

## Contributing with a new press

We welcome contributions! If you want to implement a new press, open an issue or a pull request. Refer to the [new_press.ipynb](notebooks/new_press.ipynb) notebook for a step-by-step guide to understand how presses work and what should be done to create a new one.

## Available presses

All current presses are training free. Several of them inherit from `ScorerPress` and rely on a score to prune the KV pairs with lowest importance:

- `RandomPress`: random score
- `KnormPress`: inverse norm of the key ([paper](https://arxiv.org/abs/2406.11430))
- `SnapKVPress`: average attention weight of the last queries ([paper](https://arxiv.org/abs/2404.14469))
- `ExpectedAttentionPress` (ours): expected attention weight during the generation phase  (see [this notebook](notebooks/expected_attention.ipynb))
- `StreamingLLMPress`: keep only the initial and recent tokens ([paper](https://arxiv.org/abs/2309.17453))
- `TOVAPress`: attention weight of the last query averaged across heads ([paper](https://arxiv.org/abs/2401.06104))
- `ObservedAttentionPress`: average attention weight observed during in pre-filling phase (similar to [H2O](https://arxiv.org/abs/2306.14048))

Some presses rely on a different logic:
- `ThinKPress`: compress the dimensions of the keys based on the channel attention score on the last queries ([paper](https://arxiv.org/pdf/2407.21018))
- `SimLayerKVPress`: identify "lazy" layers, and apply the StreamingLLM approach to them ([paper](https://arxiv.org/abs/2410.13846))

Finally we provide special presses:
- `AdaKVPress`: prune bottom scores of any `ScorerPress` but across all heads, achieving head-wise compressions (see [paper](https://arxiv.org/abs/2407.11550))
- `PerLayerCompressionPress`: compress each layer with a different compression ratio (experimental). This press can be used with any other press that allows to set a compression_ratio
- `ComposedPress`: compose multiple presses together by chaining their forward hooks
- `KeyRerotationPress`: rerotate pruned keys to have continuous RoPE embeddings. This press can be used with any other press that inherits from `ScorerPress`.

For a detailed list of existing KV cache compression methods, check [Awesome-KV-Cache-Compression](https://github.com/October2001/Awesome-KV-Cache-Compression) or [Awesome-LLM-Compression](https://github.com/HuangOwen/Awesome-LLM-Compression?tab=readme-ov-file#kv-cache-compression)

## Evaluation

See the [speed_and_memory.ipynb](notebooks/speed_and_memory.ipynb) notebook on how to measure peak memory usage and total time gain.
<img src="evaluation/assets/peak_memory_consumption.png" alt="drawing" width="450"/>


We provide a simple CLI to evaluate the performance of the different presses on several long-context datasets. 

_Average performance on the RULER dataset with 4k context length and Loogle Short Dependency QA task for 3 models and 7 presses_
![RULER](evaluation/assets/ruler_4096_average%20score.png)
![Loogle](evaluation/assets/loogle_shortdep_qa.png)

Please refer to the [evaluation](evaluation/README.md) directory for more details and results.

## KV cache quantization

We support KV cache quantization through the transformers `QuantizedCache` class (see [HF blog post](https://huggingface.co/blog/kv-cache-quantization#how-to-use-quantized-kv-cache-in-%F0%9F%A4%97-transformers)). To use it, simply pass a cache object to your pipeline:

```python
from transformers import QuantizedCacheConfig, QuantoQuantizedCache

config = QuantizedCacheConfig(nbits=4)
cache = QuantoQuantizedCache(config)

pipe(..., cache=cache)
```

By default, the `DynamicCache` is used (no quantization). 

> [!IMPORTANT]  
> To use the `QuantizedCache`, you need to install additional dependencies (e.g. `pip install optimum-quanto`).


## FAQ

<details><summary> 

### Which models are supported ? 
</summary>

Some presses depend on the model architecture (_e.g._ `ExpectedAttentionPress` and `SnapKVPress`) hence they might not work with all models. We tested support for `LlamaForCausalLM`, `MistralForCausalLM`, `Phi3ForCausalLM` and `Qwen2ForCausalLM` but many other models might be supported out of the box because their implementation is often similar in transformers.
</details>


<details> <summary> 

### What are the memory and throughput gains ?
</summary>

Memory usage should be reduced by around `compression_ratio * kv_cache_size`. As the KV cache is smaller, decoding should also be faster. You can measure peak memory usage gain and total time gain using [this notebook](notebooks/speed_and_memory.ipynb).
</details>


<details> <summary> 

### How does a press work ? </summary>

A press registers a forward hook (`press.forward_hook` method) to each attention layer during the pre-filling phase. Registration can be applied using the press as a context manager (`press.__call__` method):

```python
import torch
from transformers import AutoModelForCausalLM
from kvpress import KnormPress

device = "cuda:0"
ckpt = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(ckpt).to(device)
press = KnormPress(compression_ratio=0.4)

inputs = model.dummy_inputs["input_ids"].to(device)

with torch.no_grad():
    print(model(inputs).past_key_values[0][0].shape)
    # torch.Size([3, 8, 5, 128])
    
with torch.no_grad(), press(model):
    print(model(inputs).past_key_values[0][0].shape)
    # torch.Size([3, 8, 3, 128])
```
</details>

<details><summary> 

### Why not using model.generate ? 
</summary>

In fact you can use `model.generate` with a press by using the press as a context manager:

```python
with press(model):
    outputs = model.generate(inputs)
```

However, the `generate` method does not allow to exclude the question from the compression, which would artificially favors methods such as SnapKV. Ideally, we want a compression method that works whatever comes after the context (_e.g._ for use cases such as chat or document question answering). Finally the `generate` method does not allow to provide generation for multiple questions at once.

</details>

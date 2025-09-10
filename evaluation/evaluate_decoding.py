import torch
from transformers import pipeline
from kvpress import DecodingPress
from datasets import load_dataset
from transformers import DynamicCache
import yaml
import time
from tqdm import tqdm
import os

from kvpress import KnormPress, ExpectedAttentionPress, StreamingLLMPress, SnapKVPress, TOVAPress, AdaKVPress, QFilterPress

press_dict = {
    "knorm": DecodingPress(base_press=KnormPress(), compression_interval=256, target_size=4096),
    "adakv_expected_attention_e2": DecodingPress(base_press=AdaKVPress(ExpectedAttentionPress(epsilon=1e-2)), compression_interval=256, target_size=4096, hidden_states_buffer_size=256),
    "streaming_llm": DecodingPress(base_press=StreamingLLMPress(), compression_interval=256, target_size=4096),
    "tova": DecodingPress(base_press=TOVAPress(), compression_interval=256, target_size=4096, hidden_states_buffer_size=256),
    "qfilter": DecodingPress(base_press=QFilterPress(), compression_interval=256, target_size=4096),
    "knorm_8": DecodingPress(base_press=KnormPress(), compression_interval=256, target_size=8192),
    "streaming_llm_8": DecodingPress(base_press=StreamingLLMPress(), compression_interval=256, target_size=8192),
    "tova_8": DecodingPress(base_press=TOVAPress(), compression_interval=256, target_size=8192, hidden_states_buffer_size=256),
    "qfilter_8": DecodingPress(base_press=QFilterPress(), compression_interval=256, target_size=8192),
    "knorm_2": DecodingPress(base_press=KnormPress(), compression_interval=256, target_size=2048),
    "streaming_llm_2": DecodingPress(base_press=StreamingLLMPress(), compression_interval=256, target_size=2048),
    "tova_2": DecodingPress(base_press=TOVAPress(), compression_interval=256, target_size=2048, hidden_states_buffer_size=256),
    "qfilter_2": DecodingPress(base_press=QFilterPress(), compression_interval=256, target_size=2048),
    "none": None,
    "adakv_expected_attention_e2_2": DecodingPress(base_press=AdaKVPress(ExpectedAttentionPress(epsilon=1e-2)), compression_interval=256, target_size=2048, hidden_states_buffer_size=256),
    "adakv_expected_attention_e2_8": DecodingPress(base_press=AdaKVPress(ExpectedAttentionPress(epsilon=1e-2)), compression_interval=256, target_size=8192, hidden_states_buffer_size=256),

}


def evaluate_decoding(device, press_name):
    dataset = load_dataset("math-ai/aime25", split="test")

    # converto to pandas dataframe
    dataset = dataset.to_pandas()    

    # add empty columns
    dataset["pred_answer"] = ""
    dataset["cache_size"] = ""
    dataset["max_new_tokens"] = ""
    max_new_tokens = 32000
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    attn_implementation = "flash_attention_2"  # use "eager" for ObservedAttentionPress and "sdpa" if you can't use "flash_attention_2"
    pipe = pipeline("kv-press-text-generation", model=model_name, device=device, model_kwargs={"attn_implementation":attn_implementation, "dtype": torch.bfloat16}, torch_dtype=torch.bfloat16)

    press = press_dict[press_name]

    out_dir = f"aime25_test_predictions_{model_name.split('/')[-1]}_{press_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)

    init_time = time.time()
    print(f"Starting evaluation for {press_name}")
    for i,sample in tqdm(dataset.iterrows()):
        cache = DynamicCache()
        question = sample["problem"]
        pred_answer = pipe(" ", question=question, press=press, cache=cache, max_new_tokens=max_new_tokens)["answer"]
        dataset.at[i, "pred_answer"] = pred_answer[-2000:] if len(pred_answer) > 2000 else pred_answer
        dataset.at[i, "cache_size"] = cache.get_seq_length(0)
        dataset.at[i, "max_new_tokens"] = max_new_tokens
        
    end_time = time.time()
    print(f"Time taken: {end_time - init_time} seconds")
    # save the dataset
    dataset.to_csv(f"{out_dir}/aime25_test_predictions_{model_name.split('/')[-1]}_{press_name}.csv")
    # save hyperparameters
    with open(f"{out_dir}/aime25_test_predictions_{model_name.split('/')[-1]}_{press_name}.yaml", "w") as f:
        yaml.dump({"device": device, "press_name": press_name, "model_name": model_name, "attn_implementation": attn_implementation, "max_new_tokens": max_new_tokens, "compression_interval": getattr(press, "compression_interval", None) , "target_size": getattr(press, "target_size", None), "time_taken": end_time - init_time}, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--press_name", type=str, default="knorm")
    args = parser.parse_args()
    evaluate_decoding(args.device, args.press_name)
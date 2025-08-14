
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvpress import KnormPress

ckpt = "meta-llama/Meta-Llama-3-8B-Instruct"
device = "cuda"

model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype="auto").to(device)
model.set_attn_implementation("flash_attention_2")
tok = AutoTokenizer.from_pretrained(ckpt)
inputs = tok("Hello, how are you? bla bla how are you? this is some text lala ddd", return_tensors="pt").to(device)

with KnormPress(0.8)(model):
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)

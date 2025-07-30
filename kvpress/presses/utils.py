import torch
from typing import List

import torch
import importlib
from contextlib import contextmanager
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os
from datetime import datetime


@contextmanager
def patch_rotary_embedding(model):
    """
    A context manager to dynamically patch the `apply_rotary_pos_emb` function
    for any supported model architecture. It captures the query states after
    rotary embeddings are applied.

    Args:
        model (PreTrainedModel): The transformer model instance.

    Yields:
        list: A list that will be populated with the captured query tensors.
    """
    # 1. Dynamically find the model's specific "modeling" module
    try:
        module_path = model.__class__.__module__
        modeling_module = importlib.import_module(module_path)
        # print(f"🔍 Found modeling module: {module_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to import module for {model.__class__.__name__}: {e}")

    # 2. Check for the target function and save the original
    target_function = "apply_rotary_pos_emb"
    if not hasattr(modeling_module, target_function):
        raise AttributeError(
            f"Model architecture '{model.config.model_type}' is not supported. "
            f"The module '{module_path}' does not contain '{target_function}'."
        )
    
    original_function = getattr(modeling_module, target_function)
    
    captured_tensors = []

    # 3. Define the new wrapper function
    def patched_function(*args, **kwargs):
        # The original function returns a tuple (query_embed, key_embed)
        q_embed, k_embed = original_function(*args, **kwargs)
        
        # Capture the query tensor after RoPE is applied
        captured_tensors.append(q_embed.detach())
        
        return q_embed, k_embed

    # 4. Apply the patch
    setattr(modeling_module, target_function, patched_function)
    #print(f"Patch applied successfully to '{model.config.model_type}'.")
    
    try:
        # Yield the list to the user to collect the results
        yield captured_tensors
    finally:
        # 5. Restore the original function once the 'with' block is exited
        setattr(modeling_module, target_function, original_function)
        # print("Patch removed. Original function restored.")


def get_activations(model_name, dataset_name, text_column_name, batch_size, num_samples, q_len,  device):
    """
    Loads a Hugging Face model and dataset, extracts hidden states, and saves them to a file.
    """

    # --- Configure experiment directory ---
    # Create a timestamp for unique directory naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"experiments/{model_name.replace('/', '_')}_{dataset_name.replace('/', '_')}_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Experiment directory created: {experiment_dir}")

    # --- Dump the experiment config in the experiment directory ---
    # config.to_yaml(os.path.join(experiment_dir, "config.yaml"))

    # --- Load model and tokenizer ---
    print(f"Loading model and tokenizer for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval() 


    # Check for GPU and move model if available
    # device = config.device
    model.to(device)
    print(f"Using device: {device}")

    # --- 4. Load dataset and prepare for processing ---
    print(f"Loading dataset '{dataset_name}'...")
    dataset = load_dataset(dataset_name, split=f"train[:{num_samples}]")
            
    # Tokenize the dataset
    tokenizer.pad_token = tokenizer.eos_token
    def tokenize_function(examples):
        # Ensure the text_column_name actually exists in the examples
        if text_column_name not in examples:
            raise KeyError(f"Column '{text_column_name}' not found in dataset examples. Available columns: {examples.keys()}")
        return tokenizer(examples[text_column_name])

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask"])


    # --- 5. Forward batches and collect hidden states ---
    print("Collecting hidden states...")
    all_hidden_states = []
    batch_size = 1 # Adjust batch size as needed for your GPU/memory
    all_past_queries = []
    data_loader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=batch_size)

    with torch.no_grad(): # Disable gradient calculation for inference
        for i, batch in tqdm(enumerate(data_loader), total=num_samples, desc="Collecting hidden states"):
            if i >= num_samples: # Process a limited number of batches for demonstration
                print(f"Processed {num_samples} batches. Stopping for demonstration purposes.")
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # print(f"Input IDs shape: {input_ids.shape}")
            # print(f"Actual sequence lengths in batch: {[len(seq) for seq in batch['input_ids']]}")
            # print(f"Max sequence length in batch: {input_ids.shape[1]}")
            with patch_rotary_embedding(model) as captured_queries:
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            all_past_queries.append(torch.stack(captured_queries)) # (num_layers, num_heads, q_len, head_dim)

            # hidden_states = torch.cat(outputs.hidden_states, dim=0).cpu()
            # all_hidden_states.append(hidden_states)
            # all_past_queries.append(past_queries)
            # print(f"Processed batch {i+1}")
            # torch.save(hidden_states, os.path.join(config.experiment_dir, "hidden_states", f"hidden_states_{i}.pt"))
            # torch.save(past_queries, os.path.join(config.experiment_dir, "hidden_states", f"past_queries_{i}.pt"))
    
    return all_past_queries     

if __name__ == "__main__":
    all_past_queries = get_activations(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", dataset_name="kmfoda/booksum", text_column_name="chapter", batch_size=1, num_samples=1, q_len=10, device="cuda")
    torch.save(all_past_queries, "all_past_queries.pt")
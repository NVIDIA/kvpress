# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import contextlib
import logging
from typing import Optional, Any

import torch
from transformers import AutoModelForCausalLM, Cache, DynamicCache, Pipeline, AutoProcessor
from transformers.pipelines import PIPELINE_REGISTRY
from transformers.pipelines.base import GenericTensor

from kvpress.presses.base_press import BasePress
from kvpress.presses.finch_press import FinchPress
from kvpress.presses.key_rerotation_press import KeyRerotationPress
from kvpress.presses.observed_attention_press import ObservedAttentionPress
from kvpress.presses.per_layer_compression_press import PerLayerCompressionPress

logger = logging.getLogger(__name__)


class KVPressTextGenerationPipeline(Pipeline):
    """
    Pipeline for key-value cache compression in causal language models.
    Supports Text-only, Image+Text, and Audio+Text contexts.
    """

    def _sanitize_parameters(
        self,
        question: Optional[str] = None,
        questions: Optional[list[str]] = None,
        answer_prefix: Optional[str] = None,
        press: Optional[BasePress] = None,
        max_new_tokens: int = 50,
        max_context_length: Optional[int] = None,
        cache: Optional[Cache] = None,
        image: Optional[Any] = None,
        audio: Optional[Any] = None,
        **kwargs,
    ):
        answer_prefix = answer_prefix or ""
        postprocess_kwargs = {"single_question": questions is None}
        assert question is None or questions is None, "Either question or questions should be provided, not both."
        
        # Ensure we don't process both image and audio at the same time for now
        assert not (image is not None and audio is not None), "Provide either an image OR audio, but not both."

        questions = questions or ([question] if question else [""])
        if max_context_length is None:
            max_context_length = min(self.tokenizer.model_max_length, int(1e10))  # 1e10 to avoid overflow
        
        preprocess_kwargs = {
            "questions": questions,
            "answer_prefix": answer_prefix,
            "max_context_length": max_context_length,
            "image": image,
            "audio": audio,
        }
        forward_kwargs = {"press": press, "max_new_tokens": max_new_tokens, "cache": cache}
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(
        self,
        context: str,
        questions: list[str],
        answer_prefix: str,
        max_context_length: int,
        image: Optional[Any] = None,
        audio: Optional[Any] = None,
    ):
        """
        Apply chat template and tokenize. 
        Handles Text-only, Image inputs, or Audio inputs via AutoProcessor.
        """
        separator = "\n" + "#" * len(context)
        
        # 1. Handle Image Input
        if image is not None:
            processor = AutoProcessor.from_pretrained(self.model.name_or_path)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": context + separator},
                    ],
                }
            ]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            context_for_proc, question_suffix = prompt.split(separator)

            processed_inputs = processor(text=prompt, images=image, return_tensors="pt")
            
            # Prepare return dict
            result = {
                "context_ids": processed_inputs["input_ids"],
                "pixel_values": processed_inputs["pixel_values"],
                "image_sizes": processed_inputs.get("image_sizes"),
                "attention_mask": processed_inputs.get("attention_mask"),
            }

        # 2. Handle Audio Input
        elif audio is not None:
            processor = AutoProcessor.from_pretrained(self.model.name_or_path)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio"},
                        {"type": "text", "text": context + separator},
                    ],
                }
            ]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            context_for_proc, question_suffix = prompt.split(separator)

            processed_inputs = processor(text=prompt, audio=[audio], return_tensors="pt", padding=True)

            # Prepare return dict (Extract common audio features)
            result = {
                "context_ids": processed_inputs["input_ids"],
                "attention_mask": processed_inputs.get("attention_mask"),
                "input_features": processed_inputs.get("input_features"),
                "feature_attention_mask": processed_inputs.get("feature_attention_mask"),
            }

        # 3. Handle Text-Only Input
        else:
            if self.tokenizer.chat_template is None:
                bos_token = getattr(self.tokenizer, "bos_token", "")
                context = bos_token + context
                question_suffix = "\n"
            else:
                context = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": context + separator}],
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )
                context, question_suffix = context.split(separator)
            
            context_ids = self.tokenizer.encode(context, return_tensors="pt", add_special_tokens=False)
            
            # Truncate context for text-only path
            if context_ids.shape[1] > max_context_length:
                logger.warning(
                    f"Context length truncated from {context_ids.shape[1]} to {max_context_length} tokens."
                )
                context_ids = context_ids[:, :max_context_length]
                
            result = {"context_ids": context_ids}

        
        questions_processed = [question + question_suffix + answer_prefix for question in questions]
        
        
        if image is not None or audio is not None:
         
             question_ids_list = [
                 self.tokenizer.encode(
                    questions_processed[0], 
                    return_tensors="pt", 
                    add_special_tokens=False
                )
             ]
        else:
            question_ids_list = [
                self.tokenizer.encode(q, return_tensors="pt", add_special_tokens=False) 
                for q in questions_processed
            ]

        result["questions_ids"] = question_ids_list
        return result

    def _forward(
        self,
        input_tensors: dict[str, GenericTensor],
        max_new_tokens: int = 50,
        press: Optional[BasePress] = None,
        cache: Optional[Cache] = None,
    ):
        context_ids = input_tensors["context_ids"].to(self.model.device)
        context_length = context_ids.shape[1]

        # Extract multimodal tensors if they exist
        pixel_values = input_tensors.get("pixel_values")
        image_sizes = input_tensors.get("image_sizes")
        
        # Extract audio tensors
        input_features = input_tensors.get("input_features")
        feature_attention_mask = input_tensors.get("feature_attention_mask")
        
        attention_mask = input_tensors.get("attention_mask")

        # Move to device
        if pixel_values is not None: pixel_values = pixel_values.to(self.model.device)
        if image_sizes is not None: image_sizes = image_sizes.to(self.model.device)
        if input_features is not None: input_features = input_features.to(self.model.device)
        if feature_attention_mask is not None: feature_attention_mask = feature_attention_mask.to(self.model.device)
        if attention_mask is not None: attention_mask = attention_mask.to(self.model.device)

        # Prefilling
        if cache is None:
            cache = DynamicCache()

        ctx_manager = press(self.model) if press is not None else contextlib.nullcontext()
        with ctx_manager:
            model_call_kwargs = {
                "input_ids": context_ids,
                "past_key_values": cache,
                "output_attentions": self.output_attentions(press),
            }
            
            # Add Image args
            if pixel_values is not None:
                model_call_kwargs["pixel_values"] = pixel_values
            if image_sizes is not None:
                model_call_kwargs["image_sizes"] = image_sizes
            
            # Add Audio args
            if input_features is not None:
                model_call_kwargs["input_features"] = input_features
            # Note: feature_attention_mask is not compatible with flash_attention_2 in audio tower
            # Flash attention handles padding internally, so we skip it
            if feature_attention_mask is not None:
                model_call_kwargs["feature_attention_mask"] = feature_attention_mask

            # Add common args
            if attention_mask is not None:
                model_call_kwargs["attention_mask"] = attention_mask
                
            self.model(**model_call_kwargs) # For Qwen2Audio
            #self.model.model(**model_call_kwargs) # For Llava and Llama

            logger.debug(f"Context Length: {context_length}")
            logger.debug(f"Compressed Context Length: {cache.get_seq_length()}")

        answers = []
        for question_ids in input_tensors["questions_ids"]:
            if isinstance(press, KeyRerotationPress) or (isinstance(press, FinchPress) and press.rerotate_keys):
                context_length = cache.get_seq_length()

            answer = self.generate_answer(
                question_ids=question_ids.to(self.model.device),
                cache=cache,
                context_length=context_length,
                max_new_tokens=max_new_tokens,
            )
            answers.append(answer)

        return answers

    def output_attentions(self, press: BasePress):
        if isinstance(press, ObservedAttentionPress):
            return True
        if isinstance(press, (KeyRerotationPress, PerLayerCompressionPress)) and isinstance(
            press.press, ObservedAttentionPress
        ):
            return True
        return False

    def postprocess(self, model_outputs, single_question):
        if single_question:
            return {"answer": model_outputs[0]}
        return {"answers": model_outputs}

    def generate_answer(self, question_ids: torch.Tensor, cache: Cache, context_length: int, max_new_tokens: int) -> str:

        question_ids = question_ids.to(self.model.device)
        position_ids = torch.arange(
            context_length, context_length + question_ids.shape[1], device=self.model.device
        ).unsqueeze(0)

        cache_seq_lengths = [cache.get_seq_length(layer_idx) for layer_idx in range(len(cache))]

        outputs = self.model(
            input_ids=question_ids,
            past_key_values=cache,
            position_ids=position_ids,
            #num_logits_to_keep=1, # Not used  for Qwen2Audio
        )

        generated_ids = []
        last_logit = outputs.logits[0, -1]
        next_token = last_logit.argmax(dim=-1)
        generated_ids.append(next_token)
        current_position = position_ids[:, -1:].clone() + 1

        should_stop_token_ids = self.model.generation_config.eos_token_id
        if not isinstance(should_stop_token_ids, list):
            should_stop_token_ids = [should_stop_token_ids]

        for i in range(max_new_tokens - 1):
            token_input = generated_ids[-1].unsqueeze(0).unsqueeze(0)
            outputs = self.model(
                input_ids=token_input,
                past_key_values=cache,
                position_ids=current_position + i,
            )
            new_id = outputs.logits[0, -1].argmax()
            generated_ids.append(new_id)
            if new_id.item() in should_stop_token_ids:
                break

        generated_tensor = torch.stack(generated_ids).squeeze(-1)
        answer = self.tokenizer.decode(generated_tensor, skip_special_tokens=True)

        cache.key_cache = [
            cache.key_cache[layer_idx][:, :, :sequence_length]
            for layer_idx, sequence_length in enumerate(cache_seq_lengths)
        ]
        cache.value_cache = [
            cache.value_cache[layer_idx][:, :, :sequence_length]
            for layer_idx, sequence_length in enumerate(cache_seq_lengths)
        ]
        if hasattr(cache, "_quantized_key_cache"):
            cache._quantized_key_cache = [
                cache._quantized_key_cache[layer_idx][:, :, :sequence_length]
                for layer_idx, sequence_length in enumerate(cache_seq_lengths)
            ]
            cache._quantized_value_cache = [
                cache._quantized_value_cache[layer_idx][:, :, :sequence_length]
                for layer_idx, sequence_length in enumerate(cache_seq_lengths)
            ]

        return answer


PIPELINE_REGISTRY.register_pipeline(
    "kv-press-text-generation",
    pipeline_class=KVPressTextGenerationPipeline,
    pt_model=AutoModelForCausalLM,
)
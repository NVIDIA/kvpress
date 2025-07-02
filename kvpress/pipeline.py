# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import contextlib
import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, Cache, DynamicCache, Pipeline
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
    
    This pipeline enables efficient processing of long contexts by applying KV cache
    compression during the pre-filling phase, then generating answers using greedy
    decoding. It's particularly useful for question-answering tasks over long documents
    where the full context needs to be processed but memory constraints are a concern.
    
    The pipeline supports:
    - Single or multiple questions about the same context
    - Various compression methods (press objects)
    - Customizable generation parameters
    - Cache reuse for multiple questions
    
    Example usage:
    ```python
    from kvpress import KVPressTextGenerationPipeline, SnapKVPress
    
    pipeline = KVPressTextGenerationPipeline(model=model, tokenizer=tokenizer)
    press = SnapKVPress(compression_ratio=0.5)
    
    result = pipeline(
        context="Long document text...",
        question="What is the main topic?",
        press=press,
        max_new_tokens=100
    )
    ```
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
        **kwargs,
    ):
        """
        Sanitize the input parameters for the pipeline.
        The user can either provide a single question or a list of questions to be asked about the context.

        Parameters
        ----------
        question : str, optional
            The question to be asked about the context. Exclusive with `questions`.
        questions : list[str], optional
            A list of questions to be asked about the context. Exclusive with `question`.
        answer_prefix : str, optional
            The prefix to be added to the generated answer.
        press : BasePress, optional
            The key-value cache compression method to apply during pre-filling.
            
            This parameter accepts any KVPress compression method (subclass of BasePress)
            that defines how to reduce the size of the key-value cache. Common options include:
            
            - SnapKVPress: Uses recent attention patterns to identify important tokens
            - KnormPress: Scores tokens based on key vector norms
            - ExpectedAttentionPress: Uses expected attention patterns for scoring
            - BlockPress: Applies compression in blocks for memory efficiency
            - AdaKVPress: Adaptive head-wise compression with safeguards
            - ComposedPress: Combines multiple compression methods
            
            If None, no compression is applied and the full context is processed.
            The compression method significantly affects both memory usage and
            model performance, so choose based on your specific requirements.
        max_new_tokens : int, optional
            The maximum number of new tokens to generate for each answer.
        max_context_length : int, optional
            The maximum number of tokens in the context. By default will use the maximum length supported by the model.
        cache : Cache, optional
            The cache to use for the forward pass. Defaults to None (DynamicCache).
        **kwargs : dict
            Additional keyword arguments, currently ignored.

        Returns
        -------
        Tuple[dict, dict, dict]
            A tuple containing three dictionaries:
                - preprocess_kwargs: The keyword arguments for the preprocess function.
                - forward_kwargs: The keyword arguments for the forward function.
                - postprocess_kwargs: The keyword arguments for the postprocess function.
        """

        answer_prefix = answer_prefix or ""
        postprocess_kwargs = {"single_question": questions is None}
        assert question is None or questions is None, "Either question or questions should be provided, not both."
        questions = questions or ([question] if question else [""])
        if max_context_length is None:
            max_context_length = min(self.tokenizer.model_max_length, int(1e10))  # 1e10 to avoid overflow
        preprocess_kwargs = {
            "questions": questions,
            "answer_prefix": answer_prefix,
            "max_context_length": max_context_length,
        }
        forward_kwargs = {"press": press, "max_new_tokens": max_new_tokens, "cache": cache}
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(
        self,
        context: str,
        questions: list[str],
        answer_prefix: str,
        max_context_length: int,
    ):
        """
        Apply chat template and tokenize the context and questions for processing.
        
        This method prepares the input text for KV cache compression and text generation
        by applying the appropriate chat template (if available) and tokenizing the
        context and questions. It handles both models with and without chat templates.

        Parameters
        ----------
        context : str
            The long context text that will be compressed using the specified press method.
            This is typically a document, article, or other long-form text that contains
            the information needed to answer the questions.
        questions : list[str]
            List of questions to be asked about the context. Each question will be
            processed separately, but they can share the same compressed context for
            efficiency.
        answer_prefix : str
            Optional prefix to be added to each generated answer. This can be used
            to format the output or provide consistent answer formatting.
        max_context_length : int
            Maximum number of tokens allowed in the context. If the tokenized context
            exceeds this length, it will be truncated to fit within the limit.

        Returns
        -------
        dict[str, GenericTensor]
            A dictionary containing the tokenized inputs:
            - "context_ids": Tokenized context ready for compression
            - "questions_ids": List of tokenized questions for generation
            
        Notes
        -----
        The method automatically handles different tokenizer configurations:
        - For models with chat templates: Applies the template with proper formatting
        - For models without chat templates: Uses simple BOS token prefixing
        - Ensures proper separation between context and questions
        """

        # Apply chat template if available
        if self.tokenizer.chat_template is None:
            bos_token = getattr(self.tokenizer, "bos_token", "")
            context = bos_token + context
            question_suffix = "\n"  # to separate the question from the answer
        else:
            separator = "\n" + "#" * len(context)
            context = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": context + separator}], add_generation_prompt=True, tokenize=False
            )
            context, question_suffix = context.split(separator)

        # Add question_suffix and answer prefix
        # e.g. for llama3.1, question_suffix="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        questions = [question + question_suffix + answer_prefix for question in questions]

        # Tokenize the context and questions
        context_ids = self.tokenizer.encode(context, return_tensors="pt", add_special_tokens=False)
        question_ids = [
            self.tokenizer.encode(question, return_tensors="pt", add_special_tokens=False) for question in questions
        ]

        # Truncate context
        if context_ids.shape[1] > max_context_length:
            logger.warning(
                f"Context length has been truncated from {context_ids.shape[1]} to {max_context_length} tokens."
            )
            context_ids = context_ids[:, :max_context_length]

        return {"context_ids": context_ids, "questions_ids": question_ids}

    def _forward(
        self,
        input_tensors: dict[str, GenericTensor],
        max_new_tokens: int = 50,
        press: Optional[BasePress] = None,
        cache: Optional[Cache] = None,
    ):
        """
        Execute the core KV cache compression and text generation pipeline.
        
        This method performs the main processing steps: context compression using
        the specified press method, followed by greedy decoding to generate answers
        for each question. The compression is applied only during the pre-filling
        phase, while generation uses the compressed cache.

        Parameters
        ----------
        input_tensors : dict[str, GenericTensor]
            Dictionary containing tokenized inputs from the preprocess method:
            - "context_ids": Tokenized context tensor for compression
            - "questions_ids": List of tokenized question tensors for generation
        max_new_tokens : int, default=50
            Maximum number of new tokens to generate for each answer. Controls
            the length of generated responses. Larger values allow longer answers
            but increase computation time.
        press : BasePress, optional
            The compression method to apply during context pre-filling. If None,
            no compression is applied and the full context is used. The press
            method determines how the KV cache is compressed to save memory.
        cache : Cache, optional
            Pre-existing cache object to use for the forward pass. If None,
            a new DynamicCache is created. Reusing cache objects can be more
            efficient when processing multiple questions on the same context.

        Returns
        -------
        list[str]
            List of generated answers corresponding to each input question.
            The answers are generated using greedy decoding from the compressed
            context representation.
            
        Notes
        -----
        The method follows this processing pipeline:
        1. Pre-fill the context with compression applied (if press is specified)
        2. Log compression statistics (original vs compressed context length)
        3. Generate answers for each question using greedy decoding
        4. Handle special cases for methods requiring key rerotation
        5. Return the list of generated text answers
        """

        context_ids = input_tensors["context_ids"].to(self.model.device)
        context_length = context_ids.shape[1]

        # Prefilling using the press on the context
        if cache is None:
            cache = DynamicCache()

        with press(self.model) if press is not None else contextlib.nullcontext():
            self.model(
                input_ids=context_ids,
                past_key_values=cache,
                output_attentions=self.output_attentions(press),
                num_logits_to_keep=1,
            )

        logger.debug(f"Context Length: {context_length}")
        logger.debug(f"Compressed Context Length: {cache.get_seq_length()}")

        # Greedy decoding for each question
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

    def generate_answer(
        self, question_ids: torch.Tensor, cache: Cache, context_length: int, max_new_tokens: int
    ) -> str:
        """
        Generate an answer to a question using greedy decoding.

        Parameters
        ----------
        question_ids : torch.Tensor
            The tokenized question.
        cache : Cache
            The compressed key-value cache.
        context_length : int
            The length of the context.
        max_new_tokens : int
            The maximum number of new tokens to generate.

        Returns
        -------
        str
            The generated answer.
        """

        cache_seq_lengths = [cache.get_seq_length(layer_idx) for layer_idx in range(len(cache))]
        position_ids = torch.arange(
            context_length, context_length + question_ids.shape[1], device=self.model.device
        ).unsqueeze(0)

        # if the user doesn't provide a question, skip forward pass
        outputs = self.model(
            input_ids=question_ids.to(self.model.device),
            past_key_values=cache,
            position_ids=position_ids,
            num_logits_to_keep=1,
        )

        position_ids = position_ids[:, -1:] + 1
        generated_ids = [outputs.logits[0, -1].argmax()]

        should_stop_token_ids = self.model.generation_config.eos_token_id
        if not isinstance(should_stop_token_ids, list):
            should_stop_token_ids = [should_stop_token_ids]

        for i in range(max_new_tokens - 1):
            outputs = self.model(
                input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0),
                past_key_values=cache,
                position_ids=position_ids + i,
            )
            new_id = outputs.logits[0, -1].argmax()
            generated_ids.append(new_id)
            if new_id.item() in should_stop_token_ids:
                break
        answer = self.tokenizer.decode(torch.stack(generated_ids), skip_special_tokens=True)

        # Remove the generated tokens from the cache
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

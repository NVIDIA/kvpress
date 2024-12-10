# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

import torch
from torch import nn
from transformers import LlamaForCausalLM, MistralForCausalLM, Phi3ForCausalLM, PreTrainedModel, Qwen2ForCausalLM

logger = logging.getLogger(__name__)


@dataclass
class BasePress:
    """
    Base class for all pruning methods.
    The `forward_hook` method is called after the forward pass of an attention layer.
    Any pruning/updating method should be implemented in the derived class.
    """

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """Cache compression hook called after the forward pass of an attention layer.
        The hook is applied only during the pre-filling phase if there is some pruning ratio.

        Parameters
        ----------
        module :
            Transformer attention layer.
        input :
            Input to the hook. This is the input to the forward pass of the layer.
        kwargs :
            Keyword arguments, as given to the forward pass of the layer.
        output :
            Output of the hook. This is the original output of the forward pass of the layer.

        Returns
        -------
            Modified output of the forward pass of the layer.
        """
        raise NotImplementedError("forward_hook method must be implemented in the derived class")

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        """
        Context manager to apply a compression method to a model.
        Apply this context manager during the pre-filling phase to compress the context.

        Parameters
        ----------
        model : PreTrainedModel
            Model to apply the compression method to
        """

        if not isinstance(model, (LlamaForCausalLM, MistralForCausalLM, Phi3ForCausalLM, Qwen2ForCausalLM)):
            logger.warning(f"Model {type(model)} not tested")

        hooks = []
        try:
            for layer in model.model.layers:
                hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))

            yield
        finally:
            for forward_hook in hooks:
                forward_hook.remove()

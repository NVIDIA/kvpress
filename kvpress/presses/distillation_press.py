# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
from torch import nn
from torch.optim import AdamW
import torch.nn.functional as F

from kvpress.presses.expected_attention_press import ExpectedAttentionPress


class EinSumLayer(nn.Module):
    def __init__(self, weight, equation):
        super(EinSumLayer, self).__init__()
        self.equation = equation
        self.weight = nn.Parameter(weight)

    def forward(self, x):
        return torch.einsum(self.equation, self.weight, x)


class TwoLayerMLP(nn.Module):
    """
    Two-layer MLP to mimic the attention mechanism: A = softmax(Q @ K^T / sqrt(d)) @ V
    Here K and V are weights with shape (batch_size, num_key_value_heads, n_tokens, head_dim)
    """

    def __init__(self, keys, values):
        super().__init__()
        self.scale = 1 / math.sqrt(keys.shape[-1])
        self.fc_k = EinSumLayer(keys, "bhnd,bhqd->bhqn")
        self.fc_v = EinSumLayer(values, "bhnd,bhqn->bhqd")

    def forward(self, queries):
        x = self.fc_k(queries)  # Q @ K^T
        x = x * self.scale  # Q @ K^T / sqrt(d)
        x = F.softmax(x, dim=-1, dtype=torch.float32).type_as(x)  # softmax(Q @ K^T / sqrt(d))
        x = self.fc_v(x)  # softmax(Q @ K^T / sqrt(d)) @ V
        return x


class DistillationTrainer(nn.Module):
    """
    Distillation trainer to finetune the compressed keys and values to
    minimize the L2 loss between the original and compressed attention.
    """

    def __init__(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        compressed_keys: torch.Tensor,
        compressed_values: torch.Tensor,
        mean_query: torch.Tensor,
        cov_query: torch.Tensor,
    ):
        super().__init__()
        self.mean_query = mean_query
        self.L_query = torch.linalg.cholesky_ex(cov_query.float().cpu()).L.type_as(mean_query)
        self.num_key_value_heads = keys.shape[1]
        self.num_groups = mean_query.shape[1] // keys.shape[1]

        # Create MLPs
        self.mlp = TwoLayerMLP(keys, values)
        self.mlp_c = TwoLayerMLP(compressed_keys, compressed_values)

        # Deactivate gradients for original cache
        for param in self.mlp.parameters():
            param.requires_grad = False

        self.criterion = nn.MSELoss()

    def forward(self, queries):
        z = self.mlp(queries)
        zc = self.mlp_c(queries)
        loss = self.criterion(z, zc)
        return loss

    def sample_queries(self, batch_size):
        """
        We sample queries using from a Gaussian distribution with mean mean_query and covariance cov_query.
        To do this, we first sample queries from a standard normal distribution, multiply them by the Cholesky
        decomposition of the covariance matrix (cov_query = L_query @ L_query^T), and add the mean.
        """
        bsz, num_heads, d = self.mean_query.shape
        # Sample queries using the estimated mean and covariance
        q = torch.randn(batch_size, bsz, num_heads, d, device=self.mean_query.device, dtype=self.mean_query.dtype)
        q = torch.einsum("qbhi,bhji->qbhj", q, self.L_query) + self.mean_query

        # Reshape queries to take into account Grouped Query Attention
        q = q.view(batch_size, bsz, self.num_key_value_heads, self.num_groups, d)
        q = q.transpose(1, 3).reshape(batch_size * self.num_groups, bsz, self.num_key_value_heads, d)
        q = q.transpose(0, 2)
        return q

    def train(self, num_steps=100, batch_size=32, lr=0.05):
        optimizer = AdamW(self.mlp_c.parameters(), lr=lr)
        losses = []
        for _ in range(num_steps):
            q = self.sample_queries(batch_size)
            loss = self.forward(q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses


class DistillationPress(ExpectedAttentionPress):
    """
    This press adds a distillation step to the ExpectedAttentionPress. Attention is defined as
    A = softmax(Q @ K^T / sqrt(d)) @ V, where Q, K, V are the queries, keys and values. After
    compressing K and V to K_c and V_c, the distillation step finetunes their values to minimize
    the L2 loss between A and A_c. We estimate Q using the mean and covariance of the queries.

    Note that this press is significantly slower than all the other ones as it requires training.
    The training losses are stored in the losses attribute.
    """

    use_gaussian_mixture_statistics = True

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        cache = output[-1]

        # Get uncompressed keys and values
        keys = torch.clone(cache.key_cache[module.layer_idx])
        values = torch.clone(cache.value_cache[module.layer_idx])

        # Compress keys and values using ExpectedAttentionPress
        super().forward_hook(module, input, kwargs, output)
        compressed_keys = cache.key_cache[module.layer_idx]
        compressed_values = cache.value_cache[module.layer_idx]

        # Distill the keys and values into their compressed versions. cache will be automatically updated
        mean_query, cov_query = self.get_query_statistics(module, kwargs["hidden_states"])
        trainer = DistillationTrainer(keys, values, compressed_keys, compressed_values, mean_query, cov_query)
        with torch.enable_grad():
            losses = trainer.train()

        # Store losses
        if module.layer_idx == 0:
            self.losses = [losses]
        else:
            self.losses.append(losses)

        return output

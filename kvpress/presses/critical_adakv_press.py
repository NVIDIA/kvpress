from dataclasses import dataclass

import torch
from transformers.models.llama.modeling_llama import repeat_kv
from kvpress.presses.base_press import BasePress
# from kvpress.presses.adakv_press import AdaKVPress
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class CriticalAdaKVPress(BasePress):
    """
    AdaKV (https://arxiv.org/abs/2407.11550) selects the top-k keys and values among all heads in a layer
    based on the scores, achieving head-specific compression.
    A safeguard is applied to ensure a minimum fraction of KV pairs per head (alpha_safeguard parameter)
    This press has been reviewed by Yuan Feng, first author of AdaKV.
    """

    press: ScorerPress
    alpha_safeguard: float = 0.20
    epsilon: float = 1e-4
    first_stage_ratio: float = 0.5

    def __post_init__(self):
        assert 0 <= self.alpha_safeguard <= 1, "alpha_safeguard should be in [0, 1]"

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.compression_ratio == 0:
            return keys, values

        assert module.config._attn_implementation != "eager", "eager mode not supported"

        # Compute scores
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)
        bsz, num_key_value_heads, q_len = scores.shape

        # Make sure to keep at least alpha * (1 - compression_ratio) KV pairs per head
        n_kept = int(q_len * (1 - self.compression_ratio))  # ScorerPress definition
        n_safe = int(n_kept * self.alpha_safeguard)

        top_indices = torch.topk(scores, n_safe, dim=-1).indices
        scores.scatter_(-1, top_indices, torch.finfo(scores.dtype).max)

        # budget allocation
        top_indices = torch.topk(scores, n_safe, dim=-1).indices
        budget_scores = scores.scatter(-1, top_indices, torch.finfo(scores.dtype).max)
        budget_scores = budget_scores.reshape(bsz, -1)
        top_indices = torch.topk(budget_scores, n_kept * num_key_value_heads, dim=-1).indices
        top_indices_head_idx = top_indices // q_len
        head_budgets = torch.zeros(num_key_value_heads, device=keys.device, dtype=torch.int64)
        head_budgets.scatter_add_(0, top_indices_head_idx.flatten(), torch.ones_like(top_indices_head_idx.flatten()))

        # stage 1 selection
        head_selection_budget_1st = (head_budgets * self.first_stage_ratio).to(torch.int64).tolist()
        top_k_index = torch.topk(scores, max(head_selection_budget_1st), sorted=True, dim=-1).indices

        # mask for each head
        for head_idx in range(num_key_value_heads):
            phase1_budget = head_selection_budget_1st[head_idx]
            scores[:, head_idx ,:].scatter_(-1, top_k_index[ :,head_idx, :phase1_budget], torch.finfo(scores.dtype).max)

        def vwl1norm(v, w):
            '''
            v.shape = (bs, head_num, seq_len, head_dim)
            w.shape = (head_num, head_dim, dim)
            vw_norm.shape = (bs, head_num, seq_len, 1)
            '''
            v = torch.abs(v)
            w = torch.abs(w)
            return torch.einsum("abcd,bde->abc", [v, w])

        # calculating projected value norm for critical cache selection
        output_w = module.o_proj.weight.transpose(0, 1)
        head_o_proj = output_w.view(module.config.num_attention_heads, module.config.head_dim, module.config.hidden_size)
        values_states = repeat_kv(values, module.num_key_value_groups)
        projected_norm = vwl1norm(values_states, head_o_proj)
        projected_norm = projected_norm.view(bsz, num_key_value_heads, module.num_key_value_groups, q_len).mean(dim=2)

        # stage 2 selection
        scores =  (scores + self.epsilon) * projected_norm
        top_k_index = torch.topk(scores, max(head_budgets), sorted=True, dim=-1).indices
        for head_idx in range(num_key_value_heads):
            head_budget = head_budgets[head_idx]
            scores[:, head_idx ,:].scatter_(-1, top_k_index[ :,head_idx, :head_budget], torch.finfo(scores.dtype).max)

        # Compute bottom-k across heads
        n_pruned = num_key_value_heads * (q_len - n_kept)
        indices = torch.topk(-scores.reshape(bsz, -1), n_pruned, dim=1).indices.flatten()

        # Save indices to mask during the attention mechanism. Please refer to attention_patch.py for more details
        batch_indices = torch.arange(bsz).repeat_interleave(n_pruned)
        head_indices = indices // q_len
        seq_indices = indices % q_len
        module.masked_key_indices = (batch_indices, head_indices, seq_indices)
        return keys, values

# vnorm_scaler_press.py
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import pipeline

from kvpress import BasePress, ExpectedAttentionPress, ScorerPress


@dataclass
class VnormScalerPress(BasePress):
    press: ScorerPress

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), "VnormScalerPress requires a ScorerPress as input"

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        # keys.shape = values.shape = (bsz, num_heads, seq_len, hidden_dim)
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)
        v_norm = values.norm(dim=-1)
        scores = scores * v_norm
        return scores

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.compression_ratio == 0:
            return keys, values

        # Compute scores
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

        # Get indices of KV pairs with the lowest scores
        q_len = hidden_states.shape[1]
        n_kept = int(q_len * (1 - self.compression_ratio))
        indices = scores.topk(n_kept, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        # Prune keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        return keys, values


if __name__ == "__main__":
    pipe = pipeline(
        "kv-press-text-generation",
        model="h2oai/h2o-danube3-500m-chat",
        device=0 if torch.cuda.is_available() else -1,
    )

    context = "A very long text you want to compress once and for all"
    question = "\nA question about the compressed context"  # optional

    press = VnormScalerPress(press=ExpectedAttentionPress(0.5))
    answer = pipe(context, question=question, press=press)["answer"]
    print(answer)

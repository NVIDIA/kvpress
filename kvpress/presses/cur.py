import torch
import torch.nn as nn
import torch.nn.functional as F
from kvpress import BasePress, ScorerPress
from dataclasses import dataclass

def approximate_leverage_scores_matrix(X: torch.Tensor):
    scores = torch.sum(X ** 2, dim=-1)
    
    return scores

def approximate_local_leverage_scores_matrix(X: torch.Tensor, window_size:int = 16):
    padding_length = (X.shape[2]//window_size + 1)*window_size - X.shape[2]
    X_ = torch.zeros(X.shape[0], X.shape[1], X.shape[2]+padding_length, X.shape[3]).to(X.device)
    X_[:,:,-X.shape[2]:,:] = X
    
    X_ = X_.reshape(X_.shape[0], X_.shape[1], X_.shape[2]//window_size, window_size, X_.shape[3])

    scores = torch.sum(X_ ** 2, dim=-1)
    scores = scores / scores.sum(dim=-1,keepdim=True)

    scores = scores.reshape(scores.shape[0], scores.shape[1], -1)
    scores = scores[:,:,-X.shape[2]:]

    return scores

def combined_leverage_scores(A: torch.Tensor, B: torch.Tensor, r: int = 20, method='value', random=True, local_window_size: int = 16, use_local_approximation: bool = True):
    # Leverage scores on columns of A
    d = A.shape[-1]

    A = A.float()
    B = B.float()
    
    G = torch.randn(d, r, device=A.device) / torch.sqrt(torch.tensor(r, dtype=torch.float32, device=A.device))

    if random:
        col_scores_A = approximate_leverage_scores_matrix(A@G)
        row_scores_B = approximate_leverage_scores_matrix(B@G)

        if use_local_approximation:
            col_scores_A = approximate_local_leverage_scores_matrix(A@G, window_size=local_window_size)
            row_scores_B = approximate_local_leverage_scores_matrix(B@G, window_size=local_window_size)
    else:
        col_scores_A = approximate_leverage_scores_matrix(A)
        row_scores_B = approximate_leverage_scores_matrix(B)

        if use_local_approximation:
            col_scores_A = approximate_local_leverage_scores_matrix(A, window_size=local_window_size)
            row_scores_B = approximate_local_leverage_scores_matrix(B, window_size=local_window_size)

    if method == 'key':
        combined = col_scores_A
    elif method == 'value':
        combined = row_scores_B
    elif method == 'kv_avg':
        combined = 0.5 * (col_scores_A + row_scores_B)
    elif method == 'kv_product':
        combined = col_scores_A * row_scores_B
    else:
        raise ValueError("Unknown method: choose from 'kv_avg', 'key', 'value' or 'kv_product'")

    return combined / combined.sum(dim=-1, keepdim=True) # Normalize to form probability distribution

@dataclass
class CURPress(ScorerPress):
    """
    Base class for all KV cache compression methods.
    The `forward_hook` method is called after the forward pass of an attention layer to update the cache.
    """
    
    compression_ratio: float
    num_sinks: int = 4
    leverage_type: str = 'kv_product'
    use_random_leverage: bool = False
    use_local_approximation: bool = True
    local_window_size: int = 16
    
    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The core logic of the compression method.

        Parameters
        ----------
        module :
            Transformer layer, see `hook` method for more details
        hidden_states :
            Hidden states of the layer
        keys :
            Keys of the cache (unquantized)
        values :
            Values of the cache (unquantized)
        attentions :
            Attention weights of the layer
        kwargs :
            Keyword arguments, as given to the forward pass of the layer

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated keys and values
        """
        assert self.leverage_type in ["key", "value", "kv_avg", "kv_product"]

        leverage_scores = combined_leverage_scores(keys, values, r=20, method=self.leverage_type, random=self.use_random_leverage, \
                                                   local_window_size=self.local_window_size, use_local_approximation=self.use_local_approximation)
        leverage_scores[:,:,:self.num_sinks] = 1
        
        return leverage_scores

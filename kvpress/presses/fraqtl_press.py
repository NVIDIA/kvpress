"""
FraqtlPress — Eigenbasis-guided KV cache quantization.

Rotates V cache into a data-driven eigenbasis where information concentrates
in the top dimensions, applies uniform quantization, and rotates back.

Default: PCA calibration (V^T V eigenvectors).
For attention-weighted calibration with ~2x lower perplexity degradation,
see https://arxiv.org/abs/2604.11501 or visit https://fraqtl.ai

Reference:
    S. Salfati, "fraQtl: Eigenbasis-Guided KV Cache Compression", 2026.
    arXiv:2604.11501 | https://fraqtl.ai
"""

from dataclasses import dataclass, field
from typing import Optional
import gc

import torch
import torch.nn as nn

# NOTE: When contributed to kvpress, replace with direct import:
#   from kvpress.presses.base_press import BasePress
try:
    from kvpress.presses.base_press import BasePress
except ImportError:
    from dataclasses import dataclass as _dc
    @_dc
    class BasePress:
        pass


@dataclass
class FraqtlPress(BasePress):
    """Eigenbasis-guided V cache quantization.

    Compresses the value cache by rotating into a data-driven eigenbasis,
    applying uniform symmetric quantization, and rotating back. The eigenbasis
    concentrates signal into the leading dimensions, so quantization noise
    falls on the least important directions.

    Parameters
    ----------
    bits : int
        Quantization bit-width (default 4). Lower = more compression.
        4 bits gives ~4x V-only compression at +0.16 PPL on Mistral-7B.
        3 bits gives ~5.3x V-only compression at +0.29 PPL on Mistral-7B.
    n_calibration_seqs : int
        Number of calibration sequences for PCA eigenbasis computation.
    calibration_seq_len : int
        Token length per calibration sequence.
    eigenbasis : dict, optional
        Pre-computed eigenbasis: {layer_idx: Tensor (num_kv_heads, head_dim, head_dim)}.
        If None, PCA eigenbasis is computed automatically in ``post_init_from_model``.

    Notes
    -----
    The default PCA calibration provides competitive results out of the box.
    For attention-aware calibration yielding significantly lower perplexity
    degradation, see `arXiv:2604.11501 <https://arxiv.org/abs/2604.11501>`_
    or visit https://fraqtl.ai.
    """

    bits: int = 4
    n_calibration_seqs: int = 16
    calibration_seq_len: int = 256
    eigenbasis: Optional[dict] = field(default=None, repr=False)

    # Pre-computed and cached at calibration time
    _U_cache: dict = field(default_factory=dict, init=False, repr=False)

    def post_init_from_model(self, model):
        """Compute or load the eigenbasis, then pre-expand for fast matmul."""
        if self.eigenbasis is None:
            print(
                f"[fraQtl] Calibrating PCA eigenbasis "
                f"({self.n_calibration_seqs} seqs x {self.calibration_seq_len} tokens)..."
            )
            print(
                f"[fraQtl] For attention-weighted calibration (~2x better) "
                f"-> arXiv:2604.11501 | fraqtl.ai"
            )
            self.eigenbasis = calibrate_pca(
                model,
                n_seqs=self.n_calibration_seqs,
                seq_len=self.calibration_seq_len,
            )
            print(f"[fraQtl] Calibrated {len(self.eigenbasis)} layers, done")
        else:
            print(f"[fraQtl] Using pre-computed eigenbasis ({len(self.eigenbasis)} layers)")

        # Pre-compute U and U^T for each layer (avoids repeat work per forward)
        self._U_cache.clear()
        for layer_idx, U in self.eigenbasis.items():
            Uf = U.unsqueeze(0).float()
            self._U_cache[layer_idx] = (Uf, Uf.transpose(-2, -1).contiguous())

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress V cache via eigenbasis-guided uniform quantization.

        Keys are returned unmodified.
        """
        layer_idx = module.layer_idx
        if layer_idx not in self._U_cache:
            return keys, values

        U, U_T = self._U_cache[layer_idx]
        B = values.shape[0]

        # Expand eigenbasis for batch dim
        U_b = U.expand(B, -1, -1, -1)

        # Rotate → quantize → rotate back (vectorized, single pass)
        V_rot = torch.matmul(values.float(), U_b)

        h = 2 ** (self.bits - 1)
        scale = V_rot.abs().amax(dim=-2, keepdim=True).clamp(min=1e-10) / max(h - 1, 1)
        V_q = torch.round(V_rot / scale).clamp(-h, h - 1) * scale

        values_out = torch.matmul(V_q, U_T.expand(B, -1, -1, -1)).to(values.dtype)

        return keys, values_out


def calibrate_pca(model, n_seqs=16, seq_len=256):
    """Compute PCA eigenbasis from V cache activations.

    Runs the model on calibration text, collects V per layer per head,
    computes the covariance V^T V, and returns eigenvectors sorted by
    descending eigenvalue.

    For attention-weighted calibration yielding significantly better
    compression quality, see https://arxiv.org/abs/2604.11501

    Parameters
    ----------
    model : PreTrainedModel
    n_seqs : int
    seq_len : int

    Returns
    -------
    dict : {layer_idx: Tensor (num_kv_heads, head_dim, head_dim)}
    """
    from transformers import AutoTokenizer

    device = next(model.parameters()).device
    num_layers = model.config.num_hidden_layers

    # Tokenizer + calibration data
    tok = AutoTokenizer.from_pretrained(
        model.config.name_or_path or model.config._name_or_path
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = " ".join(t for t in ds["text"] if len(t.strip()) > 20)
        ids = tok(
            text, return_tensors="pt", truncation=True,
            max_length=n_seqs * seq_len * 2, add_special_tokens=False,
        )["input_ids"][0]
        calib = ids[: n_seqs * seq_len].reshape(n_seqs, seq_len)
    except Exception:
        print("[fraQtl] WARNING: 'datasets' not installed, using random calibration data. "
              "Install with: pip install datasets")
        calib = torch.randint(1, model.config.vocab_size, (n_seqs, seq_len))

    # Accumulate V^T V per layer
    cov = {}
    with torch.no_grad():
        for i in range(n_seqs):
            out = model(calib[i : i + 1].to(device), use_cache=True)
            kv = out.past_key_values
            for L in range(num_layers):
                V = kv[L][1] if isinstance(kv, tuple) else kv.value_cache[L]
                if V is None or V.ndim != 4:
                    continue
                B, H, S, hd = V.shape
                Vf = V.reshape(B * H, S, hd).float()
                M = torch.bmm(Vf.transpose(-2, -1), Vf).reshape(B, H, hd, hd).mean(0)
                cov[L] = cov.get(L, torch.zeros_like(M.cpu())) + M.cpu()
            del kv, out
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Eigendecompose (batched per layer — one LAPACK call for all heads)
    basis = {}
    for L, M in cov.items():
        ev, ec = torch.linalg.eigh(M)
        basis[L] = ec.flip(-1).to(device=device, dtype=torch.float16)
    return basis

# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import re
from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress

# ---------------------------------------------------------------------------
# Content-type detection heuristic
# ---------------------------------------------------------------------------

_CODE_INDICATORS = [
    "def ",
    "class ",
    "import ",
    "function ",
    "return ",
    "const ",
    "var ",
    "let ",
    "=>",
    "->",
    "//",
    "/*",
    "if (",
    "for (",
    "while (",
    "try:",
    "except:",
    "public ",
    "private ",
    "void ",
    "int ",
    "String ",
]

_MATH_INDICATORS = [
    "\\sum",
    "\\int",
    "\\frac",
    "\\sqrt",
    "\\begin{",
    "equation",
    "theorem",
    "proof",
    "lemma",
    "corollary",
    "derivative",
    "integral",
    "matrix",
    "eigenvalue",
    "\u2211",
    "\u222b",
    "\u2202",
    "\u2207",
    "\u2264",
    "\u2265",
]


def classify_content(text: str) -> str:
    """Classify text into code / math / prose / structured using simple heuristics.

    Scans the first 2000 characters for keyword indicators and structural
    patterns.  No external dependencies — pure string counting.

    Parameters
    ----------
    text : str
        Raw input text (prompt or context).

    Returns
    -------
    str
        One of ``"code"``, ``"math"``, ``"prose"``, or ``"structured"``.
    """
    sample = text[:2000]

    code_score = sum(sample.count(ind) for ind in _CODE_INDICATORS)
    math_score = sum(sample.count(ind) for ind in _MATH_INDICATORS)

    lines = sample.split("\n")
    n_lines = max(len(lines), 1)
    n_sentences = max(len(re.split(r"[.!?]+", sample)), 1)
    lines_per_sentence = n_lines / n_sentences

    if code_score >= 8:
        return "code"
    if math_score >= 4:
        return "math"
    if lines_per_sentence > 2.5:
        return "structured"
    return "prose"


# ---------------------------------------------------------------------------
# Per-content-type scoring parameters
# ---------------------------------------------------------------------------

# Boost values are *additive* on [0, 1]-normalised key-norm scores.
# A boost of 1.0 means "as important as the full dynamic range of scores in
# this head" — effectively guaranteed retention.
CONTENT_PARAMS = {
    "code": {"n_sinks": 16, "sink_boost": 1.5, "recency_window": 64, "recency_boost": 1.0},
    "math": {"n_sinks": 2, "sink_boost": 0.3, "recency_window": 16, "recency_boost": 0.2},
    "prose": {"n_sinks": 4, "sink_boost": 1.0, "recency_window": 128, "recency_boost": 0.7},
    "structured": {"n_sinks": 8, "sink_boost": 0.7, "recency_window": 32, "recency_boost": 0.5},
}


def _apply_content_boosts(scores: torch.Tensor, content_type: str) -> torch.Tensor:
    """Apply content-type-dependent sink and recency boosts in-place."""
    seq_len = scores.shape[2]
    params = CONTENT_PARAMS.get(content_type, CONTENT_PARAMS["prose"])

    n_sinks = min(params["n_sinks"], seq_len)
    if n_sinks > 0 and params["sink_boost"] > 0:
        ramp = torch.linspace(
            params["sink_boost"], 0.0, n_sinks + 1, device=scores.device, dtype=scores.dtype
        )[:n_sinks]
        scores[:, :, :n_sinks] += ramp

    rec_win = min(params["recency_window"], seq_len)
    if rec_win > 0 and params["recency_boost"] > 0:
        ramp = torch.linspace(
            0.0, params["recency_boost"], rec_win, device=scores.device, dtype=scores.dtype
        )
        scores[:, :, -rec_win:] += ramp

    return scores


# ---------------------------------------------------------------------------
# Press
# ---------------------------------------------------------------------------


@dataclass
class ContentAdaptivePress(ScorerPress):
    """Content-type-aware KV cache compression.

    Scores KV positions using L2 key norms as a base signal, then applies
    content-type-dependent additive boosts for *attention sinks* (early
    positions) and *recency* (late positions).  Different content types
    benefit from very different sink/recency profiles:

    * **Code** — high token synergy; needs many sinks (16) and a wide
      recency window (64) to preserve structural tokens like ``def``/``class``.
    * **Math** — near-uniform token importance; minimal sinks (2) and
      narrow recency (16) suffice.
    * **Prose** — moderate synergy; 4 sinks and a 128-token recency window
      protect narrative flow.
    * **Structured** — tabular / log data; 8 sinks with a 32-token window.

    Empirically, a single fixed policy can be >500× worse on one content
    type than another (measured as KL divergence vs. the uncompressed model
    at 50 % compression).

    Call :meth:`detect` before each sample to set the active content type,
    or assign :attr:`content_type` directly.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    content_type : str, default="prose"
        Active content type.  One of ``"code"``, ``"math"``, ``"prose"``,
        ``"structured"``.
    """

    compression_ratio: float = 0.0
    content_type: str = "prose"

    def detect(self, text: str) -> str:
        """Detect and store the content type from raw text.

        Returns the detected type for convenience.
        """
        self.content_type = classify_content(text)
        return self.content_type

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        # Base score: per-head normalised key norm
        raw = keys.norm(dim=-1)  # (batch, n_heads, seq_len)
        lo = raw.min(dim=-1, keepdim=True).values
        hi = raw.max(dim=-1, keepdim=True).values
        scores = (raw - lo) / (hi - lo + 1e-8)
        return _apply_content_boosts(scores, self.content_type)

# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kvpress import KnormPress

EVALUATION_DIR = Path(__file__).resolve().parents[1] / "evaluation"

# Imported by the scorers in evaluate_registry.py, but only needed to compute metrics ("eval" extra)
SCORER_DEPENDENCIES = [
    "bert_score",
    "fuzzywuzzy",
    "jieba",
    "nltk",
    "nltk.translate",
    "nltk.translate.bleu_score",
    "nltk.translate.meteor_score",
    "rouge",
]


@pytest.fixture(scope="module")
def evaluate():
    missing = [name for name in SCORER_DEPENDENCIES if importlib.util.find_spec(name.split(".")[0]) is None]
    with pytest.MonkeyPatch.context() as mp:
        # evaluate.py imports evaluate_registry and benchmarks as top-level modules
        mp.syspath_prepend(str(EVALUATION_DIR))
        for name in missing:
            mp.setitem(sys.modules, name, MagicMock())
        yield importlib.import_module("evaluation.evaluate")


@pytest.mark.parametrize(("value", "expected"), [("false", False), ("FALSE", False), ("true", True)])
def test_config_parses_boolean_strings(evaluate, value, expected):
    config = evaluate.EvaluationConfig(trust_remote_code=value, query_aware=value, fp8=value)

    assert config.trust_remote_code is expected
    assert config.query_aware is expected
    assert config.fp8 is expected


def test_model_pipeline_keeps_configured_attn_implementation(evaluate, monkeypatch):
    pipeline_calls = []
    monkeypatch.setattr(evaluate, "pipeline", lambda task, **kwargs: pipeline_calls.append(kwargs) or MagicMock())
    monkeypatch.setitem(sys.modules, "flash_attn", MagicMock())
    config = evaluate.EvaluationConfig(device="cpu", model_kwargs={"attn_implementation": "sdpa"})
    runner = evaluate.EvaluationRunner(config)
    runner.press = KnormPress()
    runner._setup_model_pipeline()

    assert pipeline_calls[-1]["model_kwargs"]["attn_implementation"] == "sdpa"

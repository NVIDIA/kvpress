# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import fire
import pytest

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


@pytest.fixture
def run_cli(evaluate, monkeypatch):
    """Parses command line arguments like `python evaluate.py` and returns the config it would evaluate."""
    configs = []

    class RecordingRunner:
        def __init__(self, config):
            configs.append(config)

        def run_evaluation(self):
            pass

    monkeypatch.setattr(evaluate, "EvaluationRunner", RecordingRunner)

    def run(*args):
        fire.Fire(evaluate.CliEntryPoint, command=list(args))
        return configs[-1]

    return run


@pytest.mark.parametrize(("value", "expected"), [("false", False), ("FALSE", False), ("true", True), ("True", True)])
def test_cli_parses_boolean_strings(run_cli, value, expected):
    config = run_cli("--trust_remote_code", value, "--query_aware", value, "--fp8", value)

    assert config.trust_remote_code is expected
    assert config.query_aware is expected
    assert config.fp8 is expected


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_config_rejects_non_boolean_values(evaluate, value):
    with pytest.raises(ValueError, match="trust_remote_code"):
        evaluate.EvaluationConfig(trust_remote_code=value)

# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import fire
import pytest
import yaml
from transformers import FineGrainedFP8Config

from kvpress import KnormPress
from tests.test_evaluation_utils import WordTokenizer, haystack_df

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


@pytest.fixture
def pipeline_calls(evaluate, monkeypatch):
    """Records the keyword arguments of the transformers pipeline() calls instead of loading a model."""
    calls = []

    def fake_pipeline(task, **kwargs):
        calls.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(evaluate, "pipeline", fake_pipeline)
    return calls


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


def setup_model_pipeline(evaluate, press, **config_kwargs):
    runner = evaluate.EvaluationRunner(evaluate.EvaluationConfig(device="cpu", **config_kwargs))
    runner.press = press
    runner._setup_model_pipeline()
    return runner


@pytest.mark.parametrize("flash_attn_available", [True, False])
def test_model_pipeline_keeps_configured_attn_implementation(
    evaluate, pipeline_calls, monkeypatch, flash_attn_available
):
    monkeypatch.setitem(sys.modules, "flash_attn", MagicMock() if flash_attn_available else None)
    setup_model_pipeline(evaluate, KnormPress(), model_kwargs={"attn_implementation": "sdpa"})

    assert pipeline_calls[-1]["model_kwargs"]["attn_implementation"] == "sdpa"


def test_saved_config_can_be_read_with_safe_load(evaluate, pipeline_calls, tmp_path):
    runner = setup_model_pipeline(
        evaluate, KnormPress(), fp8=True, model_kwargs={"dtype": "auto"}, needle_depth=(10, 50)
    )
    runner.config.save_config(tmp_path / "config.yaml")
    with open(tmp_path / "config.yaml") as f:
        saved_config = yaml.safe_load(f)

    assert isinstance(pipeline_calls[-1]["model_kwargs"]["quantization_config"], FineGrainedFP8Config)
    assert runner.config.model_kwargs == {"dtype": "auto"}
    assert saved_config["model_kwargs"] == {"dtype": "auto"}
    assert saved_config["needle_depth"] == [10, 50]


def test_needle_in_haystack_ignores_fraction(evaluate, monkeypatch):
    monkeypatch.setattr(evaluate, "load_dataset", lambda *args, **kwargs: MagicMock(to_pandas=lambda: haystack_df(0)))
    config = evaluate.EvaluationConfig(
        dataset="needle_in_haystack", needle_depth=[10, 90], max_context_length=1000, fraction=0.1
    )
    runner = evaluate.EvaluationRunner(config)
    runner.pipeline = MagicMock(tokenizer=WordTokenizer())

    runner._load_and_prepare_dataset()

    assert runner.df["needle_depth"].tolist() == [10, 90]

# AGENTS.md

## Project Overview

- `kvpress` is a Python library for KV cache compression using 🤗 transformers. Read `README.md` for full project context.
- Philosophy: keep one place to compare many KV cache compression methods, make evaluation easy, and favor readability over raw speed.
- Core package code lives in `kvpress/`.
- Compression methods are implemented as "presses" in `kvpress/presses/`.
- Evaluation tooling and benchmark datasets live in `evaluation/`.
- Tests live in `tests/`.

## Environment Setup

- Package manager: `uv`. Install: `uv sync`. Activate: `source .venv/bin/activate`.

## Key Entry Points

- `KVPressTextGenerationPipeline` in `kvpress/pipeline.py` is the primary user-facing API for applying a press during generation.
- `kvpress/__init__.py`: lists all available presses.
- All presses are `@dataclass` classes inheriting from `BasePress` (`kvpress/presses/base_press.py`), and many presses inherit from `ScorerPress` (`kvpress/presses/scorer_press.py`) for score-based pruning.
- Read `BasePress` and `ScorerPress` implementations to understand the press architecture and hook mechanism.

## Style

- `make format` (isort + black), `make style` (flake8, mypy, SPDX header check).
- All Python files **must** have SPDX headers:
```python
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

## Adding or Modifying a Press

1. Create `kvpress/presses/my_press.py` as a `@dataclass` inheriting from `BasePress` (or `ScorerPress` if the press is score-based).
2. Export it in `kvpress/__init__.py` (add both the import and the `__all__` entry).
3. Add tests in `tests/default_presses.py` (shared parametrized matrix) and/or `tests/presses/` (press-specific tests). Check existing examples to decide.
4. If evaluation support is needed, add a pre-configured instance to `PRESS_REGISTRY` in `evaluation/evaluate_registry.py`.
5. Update `README.md` with press description, link to paper, and source link.
6. Run `make style` and test only new/modified tests.

## Commits

- Sign commits with DCO (`git commit -s`) as required by `CONTRIBUTING.md`.

# AGENTS.md

## Project Overview

- `kvpress` is a Python library for KV cache compression using 🤗 transformers.
- Philosophy: keep one place to compare many KV cache compression methods, make evaluation easy, and favor readability over raw speed.
- Core package code lives in `kvpress/`.
- Compression methods are implemented as "presses" in `kvpress/presses/`.
- Evaluation tooling and benchmark adapters live in `evaluation/`.
- KVzap training code lives in `kvzap/`.
- Tests live in `tests/`.

## Environment Setup

- Python: `>=3.10`.
- Package/dependency manager: `uv`.
- Local dev install:
  - `uv sync`
- Optional extras:
  - `uv sync --extra flash-attn` (flash attention support)
  - `uv sync --extra eval` (evaluation dependencies)
- Activate the venv with `source .venv/bin/activate` before running Python.

## Codebase Map

### Entry Points

- `KVPressTextGenerationPipeline` in `kvpress/pipeline.py` is the primary user-facing API. It is registered as a 🤗 transformers pipeline named `"kv-press-text-generation"` at import time.
- `kvpress/__init__.py`: public exports, and calls `patch_attention_functions()` at import time to enable head-wise compression.

### Press Architecture

All presses are `@dataclass` classes inheriting from `BasePress` (`kvpress/presses/base_press.py`). Most presses inherit from `ScorerPress` (`kvpress/presses/scorer_press.py`) and implement `score()` to assign importance per token (higher = kept); `ScorerPress.compress()` then uses topk to prune. Some presses override `compress()` directly for non-score-based logic. Wrapper presses (e.g. `AdaKVPress`, `ComposedPress`, `BlockPress`, `ChunkPress`) override `forward_hook()` to delegate to an inner press. Compression is typically applied during prefilling only, but `DecodingPress`, `PrefillDecodingPress`, and `DMSPress` also support compression during decoding.

- `BasePress.__call__` is a **context manager** that registers `forward_hook` on every `layer.self_attn` module. After the context manager exits, all hooks are removed.
- Supported models are listed in `SUPPORTED_MODELS` in `kvpress/presses/base_press.py`; other models with similar attention implementations may work.
- `compression_ratio`: float in `[0, 1)` — fraction of KV pairs to **remove** (0.0 = keep all).
- `post_init_from_model(model)`: optional hook called automatically before hooks are registered, used to initialize press-specific parameters from the model.
- `kvpress/attention_patch.py` patches `ALL_ATTENTION_FUNCTIONS` globally for head-wise masking (used by `AdaKVPress`).
- `kvpress/utils.py`: helpers for extracting keys/values from cache (quantized or not) and computing pre-RoPE query/key states.
- Tensor shapes: `keys/values` are `(batch_size, num_kv_heads, seq_len, head_dim)`, `hidden_states` are `(batch_size, seq_len, hidden_dim)`, `scores` are `(batch_size, num_kv_heads, seq_len)`.

### Evaluation

- Main CLI: `evaluation/evaluate.py` (uses `fire`). Config layering: dataclass defaults → `evaluate_config.yaml` → CLI args.
- `evaluation/evaluate_registry.py`: `PRESS_REGISTRY` (pre-configured press instances), `DATASET_REGISTRY` (HF dataset IDs), `SCORER_REGISTRY` (metric functions).
- Per-benchmark scripts in `evaluation/benchmarks/<dataset>/`.
- **Must be run from the `evaluation/` directory** (relative imports): `cd evaluation && python evaluate.py --press_name knorm --compression_ratio 0.5`

## Code Style and Conventions

- Formatter: Black (`line-length = 120`, target `py310`). Import sorter: isort. Linter: flake8 + mypy (`--check-untyped-defs`).
- `make format` (isort + black), `make style` (flake8, mypy, SPDX header check).
- All Python files **must** have SPDX headers:
```python
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```
- Prefer small, targeted edits over broad refactors unless explicitly requested.

## Testing

```bash
make test          # Full test suite with coverage
make format        # Auto-format code
make style         # Lint checks (flake8, mypy, SPDX headers)
```

- **No `conftest.py`** — fixtures are imported explicitly from `tests/fixtures.py` (session-scoped model/pipeline fixtures).
- `tests/default_presses.py`: shared press matrix (`default_presses` list) with easy/hard kwargs, including mock subclasses for presses needing external resources.
- `tests/presses/test_presses.py`: main parametrized test — runs every press × every wrapper press.
- Press-specific tests live in `tests/presses/test_<press_name>.py`; integration tests in `tests/integration/`.

## Adding or Modifying a Press

1. Create `kvpress/presses/my_press.py` as a `@dataclass` inheriting from `ScorerPress` or `BasePress`.
2. Export it in `kvpress/__init__.py` (add both the import and the `__all__` entry).
3. Add test entries to `tests/default_presses.py` (add to `default_presses` list with easy/hard kwargs). Create mock subclass if the press needs external resources.
4. Add press-specific tests in `tests/presses/` if the press has complex logic beyond what the parametrized matrix covers.
5. If evaluation support is needed, add a pre-configured instance to `PRESS_REGISTRY` in `evaluation/evaluate_registry.py`.
6. Update `README.md` with press description, link to paper, and source link.
7. Ensure `make style` passes (SPDX header, flake8, mypy).

## Commit and PR Expectations

- Keep PRs focused and include tests for behavior changes.
- Ensure style and tests pass locally before finalizing (`make style && make test`).
- Sign commits with DCO (`git commit -s`) as required by `CONTRIBUTING.md`.

## Safety and Scope

- Do not commit secrets, access tokens, or private model credentials.
- Avoid unnecessary dependency upgrades/version churn unless required for the task.
- Document behavior changes that affect public APIs (`kvpress` exports or pipeline behavior).

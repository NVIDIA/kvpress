# SPDX-FileCopyrightText: Copyright (c) 2026 Satyajeeth Suresh Kannan. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from kvpress.presses.kvzip_press import KVzipPress
from kvpress.presses.restorekv_press import RestoreKVPress

# a toy vocabulary: 0-3 sinks/prefix tokens, 10-19 words, 20-29 digits, 30-39 structure
VOCAB = {**{i: f"<s{i}>" for i in range(4)}, **{10 + i: f" word{i}" for i in range(10)}}
VOCAB.update({20 + i: str(i) for i in range(10)})
VOCAB.update(
    {
        30: ".",
        31: "\n",
        32: " (",
        33: ",",
        34: "  ",
        35: " -",
        36: "'s",
        37: " é",
        38: ")",
        39: "!",
        40: "-",
        41: "-ce",
        42: " word",
        43: ".\n",
    }
)


def decode(token_id: int) -> str:
    return VOCAB[token_id]


def _scores(n_positions: int, n_layer: int = 2, n_heads: int = 3) -> torch.Tensor:
    return torch.ones(n_layer, 1, n_heads, n_positions)


def test_repeated_structure_is_demoted_and_content_is_not():
    # prefix(2) | "1. word0\n" x 5 with distinct words, plus a needle number that repeats as a token
    ids = [0, 1]
    for i in range(5):
        ids += [20 + i, 30, 10 + i, 31]  # digit, ".", word, "\n"
    ids += [27, 27, 27, 27, 27]  # the digit 7 five times: content, never demoted
    context = torch.tensor(ids)
    score = _scores(len(ids))
    n = KVzipPress.demote_structure_scores(score, context, decode, factor=0.25, min_repeats=4, start=2)
    demoted = torch.nonzero(score[0, 0, 0] < 1).flatten().tolist()
    expected = [i for i, t in enumerate(ids) if i >= 2 and t in (30, 31)]  # "." and "\n" occur 5x each
    assert demoted == expected
    assert n == len(expected)
    assert torch.allclose(score[..., expected], torch.full((2, 1, 3, len(expected)), 0.25))
    # digits, words and the prefix are untouched, even the digit repeated 5 times
    kept = [i for i in range(len(ids)) if i not in expected]
    assert torch.all(score[..., kept] == 1)


def test_min_repeats_and_start_are_respected():
    ids = [30, 30, 30, 1, 30, 30, 30, 33, 33, 33]  # "." x3 in the region, "," x3
    context = torch.tensor(ids)
    score = _scores(len(ids))
    # region starts at 3: "." occurs 3 times there, "," 3 times -> with min_repeats=4 nothing is demoted
    assert KVzipPress.demote_structure_scores(score, context, decode, 0.5, min_repeats=4, start=3) == 0
    assert torch.all(score == 1)
    # with min_repeats=3 both are demoted, but never the positions before start
    assert KVzipPress.demote_structure_scores(score, context, decode, 0.5, min_repeats=3, start=3) == 6
    assert torch.all(score[..., :3] == 1)
    assert torch.all(score[..., 4:] == 0.5)


def test_positions_beyond_the_context_are_never_touched():
    # score_val longer than context_ids (e.g. RestoreKV appended restore tokens): tail untouched
    ids = [31] * 6
    context = torch.tensor(ids)
    score = _scores(len(ids) + 8)
    assert KVzipPress.demote_structure_scores(score, context, decode, 0.25, min_repeats=4, start=0) == 6
    assert torch.all(score[..., 6:] == 1)
    assert torch.all(score[..., :6] == 0.25)


@pytest.mark.parametrize(
    "token_id,is_content",
    [
        (10, True),
        (20, True),
        (37, True),
        (30, False),
        (31, False),
        (32, False),
        (34, False),
        (35, False),
        (36, False),
        (40, False),
        (41, False),
    ],
)
def test_token_classification(token_id, is_content):
    assert (KVzipPress._CONTENT_TOKEN.fullmatch(decode(token_id).strip()) is not None) is is_content


def test_joiners_inside_uuids_are_not_demoted_but_separators_are():
    # "8f-4a-9e" style ids: a dash glued between alphanumerics is a joiner and is never demoted, however often it recurs
    uuid = [20, 21, 40, 22, 23, 40, 24, 25, 40, 26, 27]  # digits and dashes glued together
    ids = uuid + [31] + uuid + [31] + uuid + [31] + uuid + [31]  # 4 ids on 4 lines: dash x12, newline x4
    context = torch.tensor(ids)
    score = _scores(len(ids))
    n = KVzipPress.demote_structure_scores(score, context, decode, factor=0.25, min_repeats=4, start=0)
    dashes = [i for i, t in enumerate(ids) if t == 40]
    newlines = [i for i, t in enumerate(ids) if t == 31]
    assert torch.all(score[..., dashes] == 1), "UUID dashes must not be demoted"
    assert torch.all(score[..., newlines] == 0.25) and n == len(newlines)
    # a dash+letters piece ("-ce") glued inside an id is a joiner too
    ids = [20, 41, 21] * 5
    score = _scores(len(ids))
    assert (
        KVzipPress.demote_structure_scores(score, torch.tensor(ids), decode, factor=0.25, min_repeats=4, start=0) == 0
    )
    # list markers "1." followed by " word" keep their demotion: the next token starts with whitespace
    lst = []
    for i in range(5):
        lst += [20 + i, 30, 42, 31]
    score = _scores(len(lst))
    KVzipPress.demote_structure_scores(score, torch.tensor(lst), decode, factor=0.25, min_repeats=4, start=0)
    dots = [i for i, t in enumerate(lst) if t == 30]
    assert torch.all(score[..., dots] == 0.25)
    # a token carrying a newline (".\n") between two words is a separator, not a joiner
    para = [10, 43, 11] * 5
    score = _scores(len(para))
    KVzipPress.demote_structure_scores(score, torch.tensor(para), decode, factor=0.25, min_repeats=4, start=0)
    assert torch.all(score[..., [i for i, t in enumerate(para) if t == 43]] == 0.25)
    # a dash at the very first or last position has no two neighbours and is demoted like any structure token
    edge = [40, 20, 40, 21, 40, 22, 40]
    score = _scores(len(edge))
    KVzipPress.demote_structure_scores(score, torch.tensor(edge), decode, factor=0.25, min_repeats=4, start=0)
    assert torch.all(score[..., [0, 6]] == 0.25) and torch.all(score[..., [2, 4]] == 1)


def test_factor_zero_disables_and_option_is_inherited_by_restorekv():
    score = _scores(6)
    assert KVzipPress.demote_structure_scores(score, torch.tensor([31] * 6), decode, 0.0, 4) == 0
    assert torch.all(score == 1)
    press = RestoreKVPress(kvzip_plus_normalization=True, structure_demotion=0.25)
    assert press.structure_demotion == 0.25 and press.structure_min_repeats == 4
    with pytest.raises(AssertionError):
        KVzipPress(structure_demotion=1.0)

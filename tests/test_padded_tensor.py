# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from kvpress.padded_tensor import FILL_VALUE, PaddedTensor


def test_accept_last_with_swap():
    data = torch.randn(1, 2, 6, 4)
    new_token = data[0, 0, -1, :].clone()
    lengths = torch.tensor([[3, 5]])
    pt = PaddedTensor(data, lengths)

    accepted = torch.tensor([[True, True]])
    pt.accept_last(accepted)

    assert pt.lengths.tolist() == [[4, 6]]
    torch.testing.assert_close(pt.data[0, 0, 3, :], new_token)


def test_accept_last_no_gap():
    data = torch.randn(1, 2, 6, 4)
    lengths = torch.tensor([[5, 5]])
    pt = PaddedTensor(data, lengths)

    accepted = torch.tensor([[True, True]])
    pt.accept_last(accepted)
    assert pt.lengths.tolist() == [[6, 6]]


def test_accept_last_already_full():
    data = torch.randn(1, 2, 6, 4)
    lengths = torch.tensor([[6, 6]])
    pt = PaddedTensor(data, lengths)

    accepted = torch.tensor([[True, True]])
    pt.accept_last(accepted)
    assert pt.lengths.tolist() == [[6, 6]]


def test_accept_last_mixed():
    data = torch.randn(1, 2, 6, 4)
    lengths = torch.tensor([[3, 5]])
    pt = PaddedTensor(data, lengths)

    accepted = torch.tensor([[True, False]])
    pt.accept_last(accepted)
    assert pt.lengths.tolist() == [[4, 5]]


def test_fill_padding():
    data = torch.randn(1, 2, 6, 4)
    lengths = torch.tensor([[3, 5]])
    pt = PaddedTensor(data, lengths)

    pt.fill_padding()

    assert not torch.isinf(pt.data[0, 0, :3, :]).any()
    assert (pt.data[0, 0, 3:, :] == FILL_VALUE).all()
    assert not torch.isinf(pt.data[0, 1, :5, :]).any()
    assert (pt.data[0, 1, 5:, :] == FILL_VALUE).all()


def test_fill_padding_with_custom_value():
    data = torch.randn(1, 2, 6, 4)
    lengths = torch.tensor([[3, 5]])
    pt = PaddedTensor(data, lengths)

    pt.fill_padding(0)

    assert (pt.data[0, 0, 3:, :] == 0).all()
    assert (pt.data[0, 1, 5:, :] == 0).all()


def test_valid_mask():
    data = torch.randn(1, 2, 6, 4)
    lengths = torch.tensor([[3, 6]])
    pt = PaddedTensor(data, lengths)

    mask = pt.valid_mask()
    assert mask[0, 0].tolist() == [True, True, True, False, False, False]
    assert mask[0, 1].tolist() == [True, True, True, True, True, True]


def test_shrink():
    data = torch.randn(1, 2, 8, 4)
    lengths = torch.tensor([[4, 6]])
    pt = PaddedTensor(data, lengths)

    pt.shrink()
    assert pt.data.shape == (1, 2, 6, 4)


def test_remove_last():
    data = torch.randn(1, 2, 6, 4)
    lengths = torch.tensor([[5, 6]])
    pt = PaddedTensor(data, lengths)

    pt.remove_last(torch.tensor([[True, False]]))
    assert pt.lengths.tolist() == [[4, 6]]


def test_valid_mask_include_last():
    data = torch.randn(1, 2, 6, 4)
    lengths = torch.tensor([[3, 5]])
    pt = PaddedTensor(data, lengths)

    mask = pt.valid_mask(include_last=True)
    assert mask[0, 0].tolist() == [True, True, True, False, False, True]
    assert mask[0, 1].tolist() == [True, True, True, True, True, True]


def test_clone():
    data = torch.randn(1, 2, 6, 4)
    lengths = torch.tensor([[3, 5]])
    pt = PaddedTensor(data, lengths)

    pt2 = pt.clone()
    pt2.data[0, 0, 0, 0] = 999.0
    assert pt.data[0, 0, 0, 0] != 999.0

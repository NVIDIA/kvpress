# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import unittest

from evaluation.benchmarks.utils import extract_boxed


class ExtractBoxedTest(unittest.TestCase):
    def test_extracts_nested_latex(self):
        self.assertEqual(extract_boxed(r"Answer: \boxed{\frac{1}{2}}"), r"\frac{1}{2}")

    def test_selects_first_or_last_answer(self):
        text = r"Draft: \boxed{1}. Final: \boxed{2}."
        self.assertEqual(extract_boxed(text), "1")
        self.assertEqual(extract_boxed(text, last=True), "2")

    def test_returns_none_without_complete_box(self):
        self.assertIsNone(extract_boxed("No boxed answer"))
        self.assertIsNone(extract_boxed(r"Incomplete: \boxed{\frac{1}{2}"))


if __name__ == "__main__":
    unittest.main()

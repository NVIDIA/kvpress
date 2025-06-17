# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from transformers import AutoTokenizer, AutoConfig
from kvpress.presses.base_press import SUPPORTED_MODELS


def test_tokenizer_has_bos_or_eos(tokenizer_families):  # noqa: F811
    # Test that all tokenizers have either a BOS or EOS token
    for family in tokenizer_families:
        tokenizer = AutoTokenizer.from_pretrained(family, trust_remote_code=True)
        assert tokenizer.bos_token is not None or tokenizer.eos_token is not None, (
            f"Tokenizer {family} has neither BOS nor EOS token. It will not work with FinchPress"
        )


def test_supported_models_have_tokenizers(tokenizer_families):  # noqa: F811
    # Ensure kvpress.SUPPORTED_MODELS have corresponding tokenizers being tested in the fixture.
    tokenizer_families = [AutoConfig.from_pretrained(family, trust_remote_code=True).architectures[0]
                          for family in tokenizer_families]
    for model_class in SUPPORTED_MODELS:
        assert model_class.__name__ in tokenizer_families, (
            f"Model {model_class.__name__} has no corresponding tokenizer"
        )

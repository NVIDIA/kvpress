from kvpress.presses.duo_attention_press import DuoAttentionPress, PATTERNS_DICT


def test_load_attention_pattern():
    for model_name in PATTERNS_DICT:
        model = type("model", (), {"config": type("config", (), {"name_or_path": model_name})})()
        DuoAttentionPress.load_attention_pattern(model)

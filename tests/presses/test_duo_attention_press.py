<<<<<<< HEAD
from kvpress.presses.duo_attention_press import PATTERNS_DICT, DuoAttentionPress
=======
from kvpress.presses.duo_attention_press import DuoAttentionPress, PATTERNS_DICT
>>>>>>> dbe7b42 (Add DuoAttentionPress (#50))


def test_load_attention_pattern():
    for model_name in PATTERNS_DICT:
        model = type("model", (), {"config": type("config", (), {"name_or_path": model_name})})()
        DuoAttentionPress.load_attention_pattern(model)

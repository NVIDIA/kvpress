from contextlib import contextmanager
from dataclasses import dataclass

from transformers import AutoTokenizer

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress

N_SINK_CACHE = {}


@dataclass
class ProtectChatTemplatePress(BasePress):
    """
    ProtectChatTemplatePress is a press that protects the chat template tokens from being compressed.
    It can be used with any ScorerPress as input.
    """

    press: ScorerPress = None

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), "ProtectChatTemplatePress requires a ScorerPress as input"

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)
        scores[:, :, :self.n_sink] = scores.max()
        return scores

    compress = ScorerPress.compress

    @contextmanager
    def __call__(self, model):

        ckpt = model.config._name_or_path
        if ckpt not in N_SINK_CACHE:
            content = "###########"
            tokenizer = AutoTokenizer.from_pretrained(ckpt)
            assert tokenizer.chat_template is not None, "Tokenizer must have a chat template"
            chat_template = tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False)
            # Cache the number of sink tokens in N_SINK_CACHE to avoid loading the tokenizer for each call
            N_SINK_CACHE[ckpt] = len(tokenizer.encode(chat_template.split(content)[0]))

        self.n_sink = N_SINK_CACHE[ckpt]

        with super().__call__(model):
            yield

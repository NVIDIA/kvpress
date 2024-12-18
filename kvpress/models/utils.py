import re
import inspect
import importlib
import transformers
from pathlib import Path

MODEL_NAMES = ["llama", "mistral", "phi3", "qwen2"]


def add_query_states_in_cache_kwargs():
    """
    Rewrite the attention modules in modeling_{name} to include the query_states argument in the cache_kwargs
    """

    version = transformers.__version__
    for name in MODEL_NAMES:
        module = importlib.import_module(f"transformers.models.{name}.modeling_{name}")
        pattern = r"cache_kwargs = {(.*?)\}"
        repl = r'cache_kwargs = {\1, "query_states": query_states}'

        source_code = ""
        source_code += f"# 🚨🚨🚨 This code has been automatically generated using transformers {version} 🚨🚨🚨\n\n"
        source_code += f"from transformers.models.{name}.modeling_{name} import *\n"
        source_code += "from transformers.modeling_flash_attention_utils import _flash_attention_forward\n\n"
        attention_classes = getattr(module, f"{name.upper()}_ATTENTION_CLASSES")
        for cls in attention_classes.values():
            source_code += re.sub(pattern, repl, inspect.getsource(cls)).strip() + "\n\n"

        path = Path(__file__).resolve().parent / f"modeling_{name}.py"
        path.write_text(source_code)


def update_attn_implementations():
    """
    Register the kvpress attention classes in the transformers {NAME}_ATTENTION_CLASSES dictionary
    """

    for name in MODEL_NAMES:
        transformers_module = importlib.import_module(f"transformers.models.{name}.modeling_{name}")
        kvpress_module = importlib.import_module(f"kvpress.models.modeling_{name}")

        attention_classes = getattr(transformers_module, f"{name.upper()}_ATTENTION_CLASSES")
        for key, value in attention_classes.items():
            attention_classes[key] = getattr(kvpress_module, value.__name__)


if __name__ == "__main__":
    add_query_states_in_cache_kwargs()

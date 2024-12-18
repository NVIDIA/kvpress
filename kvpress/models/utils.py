import re
import inspect
import importlib
import transformers
from pathlib import Path

MODEL_NAMES = ["llama", "mistral", "phi3", "qwen2"]


def rewrite_modeling_scripts():
    """
    Rewrite the modeling_{name}.py files to include the query_states argument in the cache_kwargs
    """

    for name in MODEL_NAMES:
        module = importlib.import_module(f"transformers.models.{name}.modeling_{name}")
        pattern = r"cache_kwargs = {(.*?)\}"
        repl = r'cache_kwargs = {\1, "query_states": query_states}'
        version = transformers.__version__
        source_code = f"# 🚨🚨🚨 This code has been automatically generated using transformers {version} 🚨🚨🚨\n"
        source_code += inspect.getsource(module)
        source_code = re.sub(pattern, repl, source_code)
        source_code = source_code.replace("from ...", "from transformers.")
        source_code = source_code.replace("from .", f"from transformers.models.{name}.")
        path = Path(__file__).resolve().parent / f"modeling_{name}.py"
        path.write_text(source_code)


def update_attn_implementations():
    """
    Register the kvpress attention classes in the {NAME}_ATTENTION_CLASSES dictionaries of the transformers models
    """

    for name in MODEL_NAMES:
        transformers_module = importlib.import_module(f"transformers.models.{name}.modeling_{name}")
        transformers_attention_classes = getattr(transformers_module, f"{name.upper()}_ATTENTION_CLASSES")

        kvpress_module = importlib.import_module(f"kvpress.models.modeling_{name}")
        kvpress_attention_classes = getattr(kvpress_module, f"{name.upper()}_ATTENTION_CLASSES")

        # Update transformers_attention_classes
        for key in transformers_attention_classes:
            transformers_attention_classes[key] = kvpress_attention_classes[key]


if __name__ == "__main__":
    rewrite_modeling_scripts()

# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible HTTP server for KVPress.

Launch once, then point the same evaluation client you use with vLLM at this
process (``--endpoint http://localhost:8000/v1`` / ``base_url=http://localhost:8000/v1``).

Example::

    python -m kvpress.serve --model Qwen/Qwen3-4B-Instruct-2507 --press knorm --compression-ratio 0.5
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from kvpress import (
    AdaKVPress,
    BlockPress,
    CAMPress,
    ChunkKVPress,
    CompactorPress,
    ComposedPress,
    CriticalAdaKVPress,
    CriticalKVPress,
    CURPress,
    DecodingPress,
    DMSPress,
    DuoAttentionPress,
    EntropyGatedChunkKVPress,
    ExpectedAttentionPress,
    FastKVzipPress,
    FinchPress,
    KeyDiffPress,
    KnormPress,
    KVComposePress,
    KVzapPress,
    KVzipPress,
    LagKVPress,
    LUKVPress,
    MergingPress,
    ObservedAttentionPress,
    PyramidKVPress,
    QFilterPress,
    RandomPress,
    RestoreKVPress,
    SnapKVPress,
    StreamingLLMPress,
    ThinKPress,
    TOVAPress,
)
from kvpress.presses.base_press import BasePress

logger = logging.getLogger("kvpress.serve")


def default_press_registry() -> dict[str, Optional[BasePress]]:
    """Press names used by evaluation; constructed fresh so the server owns the instance."""
    return {
        "adakv_snapkv": AdaKVPress(SnapKVPress()),
        "block_keydiff": BlockPress(press=KeyDiffPress(), block_size=128),
        "chunkkv": ChunkKVPress(press=SnapKVPress(), chunk_length=20),
        "critical_adakv_expected_attention": CriticalAdaKVPress(ExpectedAttentionPress(use_vnorm=False)),
        "critical_adakv_snapkv": CriticalAdaKVPress(SnapKVPress()),
        "critical_expected_attention": CriticalKVPress(ExpectedAttentionPress(use_vnorm=False)),
        "critical_snapkv": CriticalKVPress(SnapKVPress()),
        "cur": CURPress(),
        "duo_attention": DuoAttentionPress(),
        "duo_attention_on_the_fly": DuoAttentionPress(on_the_fly_scoring=True),
        "entropy_gated_chunkkv": EntropyGatedChunkKVPress(press=SnapKVPress()),
        "expected_attention": AdaKVPress(ExpectedAttentionPress(epsilon=1e-2)),
        "fastkvzip": FastKVzipPress(),
        "finch": FinchPress(),
        "keydiff": KeyDiffPress(),
        "kvcompose": KVComposePress(),
        "kvcompose_unstructured": KVComposePress(structured=False),
        "kvzip": KVzipPress(),
        "kvzip_plus": KVzipPress(kvzip_plus_normalization=True),
        "kvzap_linear": DMSPress(press=KVzapPress(model_type="linear")),
        "kvzap_mlp": DMSPress(press=KVzapPress(model_type="mlp")),
        "kvzap_mlp_head": KVzapPress(model_type="mlp"),
        "kvzap_mlp_layer": AdaKVPress(KVzapPress(model_type="mlp")),
        "lagkv": LagKVPress(),
        "lukv": LUKVPress(ExpectedAttentionPress(epsilon=2e-2), sink=4, window=1),
        "knorm": KnormPress(),
        "observed_attention": ObservedAttentionPress(),
        "pyramidkv": PyramidKVPress(),
        "qfilter": QFilterPress(),
        "random": RandomPress(),
        "RestoreKV": RestoreKVPress(),
        "RestoreKV_plus": RestoreKVPress(kvzip_plus_normalization=True),
        "snap_think": ComposedPress([SnapKVPress(), ThinKPress()]),
        "snapkv": SnapKVPress(),
        "streaming_llm": StreamingLLMPress(),
        "think": ThinKPress(),
        "tova": TOVAPress(),
        "compactor": CompactorPress(),
        "adakv_compactor": AdaKVPress(CompactorPress()),
        "no_press": None,
        "cam_streaming_llm": CAMPress(base_press=StreamingLLMPress()),
        "cam_knorm": CAMPress(base_press=KnormPress()),
        "cam_adakv_snapkv": CAMPress(base_press=AdaKVPress(SnapKVPress())),
        "cam_tova": CAMPress(base_press=TOVAPress()),
        "decoding_knorm": DecodingPress(base_press=KnormPress()),
        "decoding_streaming_llm": DecodingPress(base_press=StreamingLLMPress()),
        "decoding_tova": DecodingPress(base_press=TOVAPress()),
        "decoding_qfilter": DecodingPress(base_press=QFilterPress()),
        "decoding_adakv_expected_attention_e2": DecodingPress(
            base_press=AdaKVPress(ExpectedAttentionPress(epsilon=1e-2))
        ),
        "decoding_adakv_snapkv": DecodingPress(base_press=AdaKVPress(SnapKVPress())),
        "decoding_keydiff": DecodingPress(base_press=KeyDiffPress()),
        "merging_knorm": MergingPress(KnormPress()),
        "merging_snapkv": MergingPress(SnapKVPress()),
        "merging_expected_attention": MergingPress(ExpectedAttentionPress(epsilon=1e-2)),
        "merging_kvzap_mlp": MergingPress(KVzapPress(model_type="mlp")),
    }


def configure_press(
    press: Optional[BasePress],
    *,
    compression_ratio: Optional[float] = None,
    key_channel_compression_ratio: Optional[float] = None,
    head_compression_ratio: Optional[float] = None,
    threshold: Optional[float] = None,
    compression_interval: Optional[int] = None,
    target_size: Optional[int] = None,
    hidden_states_buffer_size: Optional[int] = None,
) -> Optional[BasePress]:
    """Apply CLI compression knobs, matching ``evaluation/evaluate.py``."""
    if press is None:
        return None
    if isinstance(press, DuoAttentionPress):
        if head_compression_ratio is None:
            raise ValueError("--head-compression-ratio is required for DuoAttentionPress")
        press.head_compression_ratio = head_compression_ratio
    elif isinstance(press, DMSPress):
        if threshold is None:
            raise ValueError("--threshold is required for DMSPress")
        press.threshold = threshold
    elif isinstance(press, ComposedPress):
        for component in press.presses:
            if isinstance(component, ThinKPress):
                if key_channel_compression_ratio is None:
                    raise ValueError("--key-channel-compression-ratio is required for ThinKPress")
                component.key_channel_compression_ratio = key_channel_compression_ratio
            elif hasattr(component, "compression_ratio") and compression_ratio is not None:
                component.compression_ratio = compression_ratio
    elif isinstance(press, ThinKPress):
        if key_channel_compression_ratio is None:
            raise ValueError("--key-channel-compression-ratio is required for ThinKPress")
        press.key_channel_compression_ratio = key_channel_compression_ratio
    elif isinstance(press, DecodingPress):
        if compression_interval is not None:
            press.compression_interval = compression_interval
        if target_size is not None:
            press.target_size = target_size
        if hidden_states_buffer_size is not None:
            press.hidden_states_buffer_size = hidden_states_buffer_size
    elif compression_ratio is not None and hasattr(press, "compression_ratio"):
        press.compression_ratio = compression_ratio
    return press


def message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text") or "")
        return "".join(parts)
    return str(content)


def normalize_messages(messages: list[dict]) -> list[dict[str, str]]:
    normalized = []
    for message in messages:
        role = message.get("role") or "user"
        normalized.append({"role": role, "content": message_content_to_text(message.get("content"))})
    return normalized


def enable_thinking_from_body(body: dict, default: bool) -> bool:
    template = body.get("chat_template_kwargs")
    if isinstance(template, dict) and "enable_thinking" in template:
        return bool(template["enable_thinking"])
    extra = body.get("extra_body")
    if isinstance(extra, dict):
        nested = extra.get("chat_template_kwargs")
        if isinstance(nested, dict) and "enable_thinking" in nested:
            return bool(nested["enable_thinking"])
    return default


def apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    enable_thinking: bool,
) -> str:
    if getattr(tokenizer, "chat_template", None):
        kwargs: dict[str, Any] = dict(tokenize=False, add_generation_prompt=True)
        try:
            return str(tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs))
        except TypeError:
            return str(tokenizer.apply_chat_template(messages, **kwargs))
    bos = getattr(tokenizer, "bos_token", "") or ""
    parts = [bos]
    for message in messages:
        parts.append(f"{message['role']}: {message['content']}\n")
    parts.append("assistant: ")
    return "".join(parts)


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


@dataclass
class KVPressEngine:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    model_id: str
    press: Optional[BasePress] = None
    max_model_len: Optional[int] = None
    default_enable_thinking: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.press is not None:
            if hasattr(self.press, "post_init_from_model"):
                self.press.post_init_from_model(self.model)
            if hasattr(self.press, "update_model_and_tokenizer"):
                self.press.update_model_and_tokenizer(self.model, self.tokenizer)

    def generate(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        enable_thinking: Optional[bool] = None,
    ) -> GenerationResult:
        thinking = self.default_enable_thinking if enable_thinking is None else enable_thinking
        prompt = apply_chat_template(self.tokenizer, normalize_messages(messages), thinking)
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(self.model.device)
        if self.max_model_len is not None and input_ids.shape[1] > self.max_model_len:
            logger.warning(
                "Prompt truncated from %s to %s tokens (keeping the end of the conversation).",
                input_ids.shape[1],
                self.max_model_len,
            )
            input_ids = input_ids[:, -self.max_model_len :]
        attention_mask = torch.ones_like(input_ids)
        prompt_tokens = int(input_ids.shape[1])

        generate_kwargs: dict[str, Any] = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max(1, int(max_tokens)),
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
            if top_p is not None:
                generate_kwargs["top_p"] = top_p

        press_ctx = self.press(self.model) if self.press is not None else contextlib.nullcontext()
        with self._lock, torch.inference_mode(), press_ctx:
            outputs = self.model.generate(**generate_kwargs)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        generated = outputs[0, prompt_tokens:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        completion_tokens = int(generated.shape[0])
        eos_id = self.tokenizer.eos_token_id
        hit_eos = False
        if eos_id is not None and completion_tokens > 0:
            eos_ids = eos_id if isinstance(eos_id, list) else [eos_id]
            hit_eos = int(generated[-1].item()) in eos_ids
        finish_reason = "stop" if hit_eos or completion_tokens < max_tokens else "length"
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )


def chat_completion_response(model_id: str, result: GenerationResult) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
    }


def create_app(engine: KVPressEngine):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
    except ImportError as exc:
        raise ImportError("Install the serve extra: `uv sync --extra serve` or `pip install kvpress[serve]`") from exc

    app = FastAPI(title="KVPress OpenAI-compatible server", version="0.1.0")
    created = int(time.time())

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    @app.get("/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.model_id,
                    "object": "model",
                    "created": created,
                    "owned_by": "kvpress",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    def chat_completions(body: dict):
        if body.get("stream"):
            raise HTTPException(status_code=400, detail="streaming is not supported")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="'messages' must be a non-empty list")
        max_tokens = body.get("max_tokens")
        if max_tokens is None:
            max_tokens = body.get("max_completion_tokens", 512)
        try:
            result = engine.generate(
                messages,
                max_tokens=int(max_tokens),
                temperature=float(body.get("temperature") or 0.0),
                top_p=body.get("top_p"),
                enable_thinking=enable_thinking_from_body(body, engine.default_enable_thinking),
            )
        except Exception as exc:
            logger.exception("Generation failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return JSONResponse(chat_completion_response(body.get("model") or engine.model_id, result))

    return app


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAI-compatible KVPress server (vLLM-eval drop-in).")
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=None, help="Left-truncate prompts longer than this")
    parser.add_argument(
        "--press",
        default="no_press",
        help="Press name from the evaluation registry (e.g. knorm, snapkv, kvzip, no_press)",
    )
    parser.add_argument("--compression-ratio", type=float, default=None)
    parser.add_argument("--key-channel-compression-ratio", type=float, default=None)
    parser.add_argument("--head-compression-ratio", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--compression-interval", type=int, default=None)
    parser.add_argument("--target-size", type=int, default=None)
    parser.add_argument("--hidden-states-buffer-size", type=int, default=None)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        help="Default chat_template_kwargs.enable_thinking if the request omits it",
    )
    thinking.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    parser.set_defaults(enable_thinking=False)
    parser.add_argument("--served-model-name", default=None, help="Id returned by /v1/models (defaults to --model)")
    return parser.parse_args(argv)


def load_engine(args: argparse.Namespace) -> KVPressEngine:
    registry = default_press_registry()
    if args.press not in registry:
        names = ", ".join(sorted(registry))
        raise SystemExit(f"Unknown press {args.press!r}. Choose from: {names}")
    press = configure_press(
        registry[args.press],
        compression_ratio=args.compression_ratio,
        key_channel_compression_ratio=args.key_channel_compression_ratio,
        head_compression_ratio=args.head_compression_ratio,
        threshold=args.threshold,
        compression_interval=args.compression_interval,
        target_size=args.target_size,
        hidden_states_buffer_size=args.hidden_states_buffer_size,
    )
    model_kwargs: dict[str, Any] = dict(
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    logger.info("Loading model %s", args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    return KVPressEngine(
        model=model,
        tokenizer=tokenizer,
        model_id=args.served_model_name or args.model,
        press=press,
        max_model_len=args.max_model_len,
        default_enable_thinking=args.enable_thinking,
    )


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install the serve extra: `uv sync --extra serve` or `pip install kvpress[serve]`") from exc
    engine = load_engine(args)
    app = create_app(engine)
    logger.info(
        "Serving %s with press=%s on http://%s:%s/v1 (OpenAI-compatible)",
        engine.model_id,
        args.press,
        args.host,
        args.port,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

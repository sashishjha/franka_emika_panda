#!/usr/bin/env python3
"""Serve OpenVLA on a ROCm/HIP GPU for the local MuJoCo policy client.

This process belongs on an RBCCPS GPU allocation, not on the MacBook.  It
accepts only an RGB frame and language instruction and returns one
already-un-normalized seven-dimensional action.  It intentionally has no
MuJoCo dependency and no access to simulator state.

The client protocol is documented in ``cluster/README.md``.  PyTorch on ROCm
uses the ``cuda`` device namespace, so ``cuda:0`` is correct when ``torch`` was
installed with a HIP/ROCm build.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class ServerConfig:
    model_path: str
    attention_implementation: Optional[str]


def _decode_rgb_image(value: Any) -> np.ndarray:
    if not isinstance(value, dict):
        raise ValueError("image must be an object encoded as raw-rgb-base64")
    if value.get("encoding") != "raw-rgb-base64":
        raise ValueError("unsupported image encoding")
    shape = value.get("shape")
    if not isinstance(shape, list) or len(shape) != 3:
        raise ValueError("image.shape must be [height, width, 3]")
    height, width, channels = shape
    if not all(isinstance(dim, int) for dim in shape) or height <= 0 or width <= 0 or channels != 3:
        raise ValueError("image.shape must contain positive height/width and exactly three channels")
    if height > 4096 or width > 4096:
        raise ValueError("image dimensions exceed the server safety limit")
    encoded = value.get("data")
    if not isinstance(encoded, str):
        raise ValueError("image.data must be a base64 string")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image.data is not valid base64") from exc
    expected_size = height * width * channels
    if len(raw) != expected_size:
        raise ValueError(f"image byte length is {len(raw)}, expected {expected_size}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, channels)


class OpenVLAPolicyServer:
    def __init__(self, config: ServerConfig) -> None:
        # Keep heavy imports here so a local source checkout can be linted and
        # inspected without installing the ROCm-only inference environment.
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError(
                "No GPU backend is available. Start this server in an RBCCPS ROCm allocation "
                "with a ROCm-enabled PyTorch installation."
            )
        self._torch = torch
        self.device = torch.device("cuda:0")
        self.processor = AutoProcessor.from_pretrained(config.model_path, trust_remote_code=True)
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if config.attention_implementation:
            load_kwargs["attn_implementation"] = config.attention_implementation
        self.model = AutoModelForVision2Seq.from_pretrained(config.model_path, **load_kwargs).to(self.device)
        self.model.eval()

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "device": str(self.device),
            "rocm_version": self._torch.version.hip,
            "cuda_version": self._torch.version.cuda,
        }

    def predict(self, payload: Dict[str, Any]) -> np.ndarray:
        from PIL import Image

        image = _decode_rgb_image(payload.get("image"))
        instruction = payload.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        unnorm_key = payload.get("unnorm_key")
        if unnorm_key is not None and not isinstance(unnorm_key, str):
            raise ValueError("unnorm_key must be a string when supplied")

        prompt = f"In: What action should the robot take to {instruction.strip().lower()}?\nOut:"
        inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(
            self.device, dtype=self._torch.bfloat16
        )
        with self._torch.inference_mode():
            action = self.model.predict_action(
                **inputs,
                unnorm_key=unnorm_key,
                do_sample=False,
            )
        action = np.asarray(action, dtype=float)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise RuntimeError(f"OpenVLA returned an invalid action with shape {action.shape}")
        return action


def create_app(server: OpenVLAPolicyServer):
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="Panda OpenVLA inference service")

    @app.get("/healthz")
    def healthz() -> Dict[str, Any]:
        return server.health()

    @app.post("/act")
    def act(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return {"action": server.predict(payload).tolist()}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"inference failed: {type(exc).__name__}: {exc}") from exc

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="openvla/openvla-7b")
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        choices=("flash_attention_2", "sdpa", "eager"),
        help="Use sdpa/eager as a compatibility fallback if FlashAttention cannot be built on the cluster.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = OpenVLAPolicyServer(ServerConfig(args.model_path, args.attn_implementation))
    app = create_app(server)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

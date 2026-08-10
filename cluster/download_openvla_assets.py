#!/usr/bin/env python3
"""Download the OpenVLA model snapshot and supporting files.

This script is the cluster-side bootstrap step for the remote OpenVLA service.
It downloads a full Hugging Face repository snapshot into a local directory so
the inference server can run offline from that path.

The downloader intentionally does not assume CUDA. It only fetches files.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


DEFAULT_MODEL_ID = "openvla/openvla-7b"
DEFAULT_REVISION = "main"
DEFAULT_OUTPUT_DIR = Path("cluster/models/openvla-7b")
DEFAULT_CACHE_DIR = Path(".hf_cache")


def _resolve_token(explicit_token: Optional[str]) -> Optional[str]:
    if explicit_token:
        return explicit_token
    for env_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(env_name)
        if token:
            return token
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. Prefer HF_TOKEN/HUGGING_FACE_HUB_TOKEN env vars.",
    )
    parser.add_argument(
        "--print-server-command",
        action="store_true",
        help="Print a ready-to-use openvla_server.py command after the download finishes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    token = _resolve_token(args.token)
    if token is None:
        print("[download] No Hugging Face token found. Public repos will work; gated repos will not.")

    print(f"[download] model_id   = {args.model_id}")
    print(f"[download] revision   = {args.revision}")
    print(f"[download] output_dir = {output_dir}")
    print(f"[download] cache_dir  = {cache_dir}")

    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Create and activate a virtual environment first, then run \"python3 -m pip install -r cluster/requirements-openvla-server.txt\" inside it."
        ) from exc

    local_path = snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        cache_dir=str(cache_dir),
        token=token,
        resume_download=True,
    )

    manifest = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "revision": args.revision,
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "local_path": local_path,
        "server_command": f"python cluster/openvla_server.py --model-path {output_dir} --host 127.0.0.1 --port 8000",
    }
    manifest_path = output_dir / "download-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"[download] completed: {local_path}")
    print(f"[download] manifest:  {manifest_path}")
    if args.print_server_command:
        print(manifest["server_command"])


if __name__ == "__main__":
    main()
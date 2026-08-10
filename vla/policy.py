"""Policy client abstractions for local mocks and remote OpenVLA inference."""

from __future__ import annotations

import base64
import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from .action_adapter import ActionValidationError


class ActionPolicy(Protocol):
    """A synchronous image-and-language policy producing one 7-D action."""

    def act(self, image: np.ndarray, instruction: str) -> np.ndarray:
        """Return an un-normalized action of shape ``(7,)``."""


@dataclass(frozen=True)
class RemotePolicyConfig:
    endpoint: str
    timeout_seconds: float = 10.0
    unnorm_key: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.endpoint.startswith(("http://", "https://")):
            raise ValueError("endpoint must start with http:// or https://")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


class RemoteOpenVLAPolicy:
    """Dependency-light client for the companion OpenVLA service.

    Frames are transported as base64 raw RGB bytes so the simulator side does
    not need PyTorch, Pillow, requests, or json-numpy.  The server validates
    the image shape and translates it into a NumPy array for OpenVLA.
    """

    def __init__(self, config: RemotePolicyConfig) -> None:
        self.config = config

    def act(self, image: np.ndarray, instruction: str) -> np.ndarray:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ActionValidationError("image must be an HxWx3 uint8 RGB array")
        if not instruction.strip():
            raise ActionValidationError("instruction must not be empty")

        payload = {
            "image": {
                "encoding": "raw-rgb-base64",
                "shape": list(image.shape),
                "data": base64.b64encode(image.tobytes()).decode("ascii"),
            },
            "instruction": instruction,
        }
        if self.config.unnorm_key:
            payload["unnorm_key"] = self.config.unnorm_key

        request = Request(
            self.config.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenVLA server returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"OpenVLA server is unreachable: {exc.reason}") from exc

        try:
            decoded = json.loads(body)
            raw_action = decoded["action"] if isinstance(decoded, dict) else decoded
        except (KeyError, json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError(f"OpenVLA server returned an invalid action payload: {body[:300]!r}") from exc
        return _validate_action(raw_action)


class ConstantPolicy:
    """A deterministic local policy used to validate the RGB/action/control path."""

    def __init__(self, action: Sequence[float]) -> None:
        self._action = _validate_action(action)

    def act(self, image: np.ndarray, instruction: str) -> np.ndarray:
        del image, instruction
        return self._action.copy()


class AsyncPolicyRunner:
    """Run a potentially slow policy without blocking the MuJoCo control loop."""

    def __init__(self, policy: ActionPolicy) -> None:
        self._policy = policy
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vla-policy")
        self._future: Optional[Future[np.ndarray]] = None
        self._lock = threading.Lock()
        self.last_error: Optional[str] = None

    @property
    def request_in_flight(self) -> bool:
        with self._lock:
            return self._future is not None and not self._future.done()

    def submit(self, image: np.ndarray, instruction: str) -> bool:
        """Schedule one inference request if no previous request is in flight."""

        with self._lock:
            if self._future is not None and not self._future.done():
                return False
            self._future = self._executor.submit(self._policy.act, image.copy(), instruction)
            return True

    def poll(self) -> Optional[np.ndarray]:
        """Return the latest completed action, without blocking."""

        with self._lock:
            future = self._future
            if future is None or not future.done():
                return None
            self._future = None
        try:
            action = _validate_action(future.result())
            self.last_error = None
            return action
        except Exception as exc:  # A remote failure must not stop physics/control.
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> "AsyncPolicyRunner":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _validate_action(action: Sequence[float]) -> np.ndarray:
    vector = np.asarray(action, dtype=float)
    if vector.shape != (7,):
        raise ActionValidationError(f"policy action must have shape (7,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ActionValidationError("policy action must contain only finite values")
    return vector

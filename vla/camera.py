"""MuJoCo RGB camera capture for policy observations.

The passive viewer is intentionally not used here: it is a human debug UI and
its free camera can change at runtime.  A named MJCF camera gives the policy a
stable, reproducible observation stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class PolicyCameraConfig:
    camera_name: str = "policy_camera"
    width: int = 224
    height: int = 224
    hide_debug_sites: bool = True

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera dimensions must be positive")


class PolicyCamera:
    """Owns a MuJoCo renderer and returns policy-safe RGB frames."""

    def __init__(self, model: mujoco.MjModel, config: PolicyCameraConfig | None = None) -> None:
        self.config = config or PolicyCameraConfig()
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self.config.camera_name)
        if camera_id < 0:
            raise ValueError(
                f"Camera {self.config.camera_name!r} is not defined in the loaded MJCF model"
            )
        self._camera_id = camera_id
        self._renderer = mujoco.Renderer(model, height=self.config.height, width=self.config.width)
        self._scene_option = mujoco.MjvOption()
        if self.config.hide_debug_sites:
            # The existing red/green/path/trail sites are privileged visual debug
            # overlays.  They must not become shortcuts in policy observations.
            self._scene_option.sitegroup[:] = 0

    def render(self, data: mujoco.MjData) -> np.ndarray:
        self._renderer.update_scene(
            data,
            camera=self._camera_id,
            scene_option=self._scene_option,
        )
        frame = self._renderer.render()
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
            raise RuntimeError(f"Expected HxWx3 uint8 RGB render, got {frame.shape} {frame.dtype}")
        return frame.copy()

    def close(self) -> None:
        self._renderer.close()

    def __enter__(self) -> "PolicyCamera":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

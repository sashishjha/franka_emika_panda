"""Translate a VLA Cartesian action into a smooth controller reference.

OpenVLA-style seven-dimensional actions are not Panda joint torques.  This
adapter treats an action as ``[dx, dy, dz, dRx, dRy, dRz, gripper]`` where the
translation is a world-frame displacement in metres.  The current controller
only tracks Cartesian position, so rotation is retained for logging but is not
commanded until an orientation controller is added deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


class ActionValidationError(ValueError):
    """Raised when a policy action does not satisfy the integration contract."""


@dataclass(frozen=True)
class ActionAdapterConfig:
    """Safety and smoothing parameters for Cartesian VLA actions.

    ``max_translation_step_m`` must match the demonstration action horizon.
    The default is suitable for the existing 0.12 m/s analytic trajectories
    sampled at 5 Hz (roughly 0.024 m per policy update).
    """

    max_translation_step_m: float = 0.03
    max_velocity_mps: float = 0.20
    max_acceleration_mps2: float = 1.50
    natural_frequency_hz: float = 5.0
    damping_ratio: float = 1.0
    gripper_open_threshold: float = 0.5
    gripper_open_ctrl: float = 255.0
    gripper_closed_ctrl: float = 0.0
    workspace_lower: Optional[Tuple[float, float, float]] = None
    workspace_upper: Optional[Tuple[float, float, float]] = None

    def __post_init__(self) -> None:
        positive = {
            "max_translation_step_m": self.max_translation_step_m,
            "max_velocity_mps": self.max_velocity_mps,
            "max_acceleration_mps2": self.max_acceleration_mps2,
            "natural_frequency_hz": self.natural_frequency_hz,
            "damping_ratio": self.damping_ratio,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")
        if (self.workspace_lower is None) != (self.workspace_upper is None):
            raise ValueError("workspace_lower and workspace_upper must be supplied together")
        if self.workspace_lower is not None:
            lower = np.asarray(self.workspace_lower, dtype=float)
            upper = np.asarray(self.workspace_upper, dtype=float)
            if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
                raise ValueError("workspace bounds must be ordered 3-D lower/upper vectors")


@dataclass(frozen=True)
class CartesianReference:
    """Reference consumed by the existing translational controller."""

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    gripper_ctrl: float
    rotation_delta: np.ndarray


def _clip_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= limit or norm == 0.0:
        return vector
    return vector * (limit / norm)


class CartesianActionAdapter:
    """Smoothly integrate world-frame VLA deltas into Cartesian references."""

    def __init__(self, config: ActionAdapterConfig | None = None) -> None:
        self.config = config or ActionAdapterConfig()
        self._position: Optional[np.ndarray] = None
        self._velocity = np.zeros(3, dtype=float)
        self._target: Optional[np.ndarray] = None
        self._gripper_ctrl = self.config.gripper_open_ctrl
        self._rotation_delta = np.zeros(3, dtype=float)

    @property
    def is_ready(self) -> bool:
        return self._position is not None

    @property
    def target_position(self) -> np.ndarray:
        self._require_ready()
        return self._target.copy()

    def reset(self, initial_position: Sequence[float], *, gripper_ctrl: float | None = None) -> None:
        position = self._as_vector(initial_position, 3, "initial_position")
        self._position = self._bound_workspace(position)
        self._target = self._position.copy()
        self._velocity = np.zeros(3, dtype=float)
        self._rotation_delta = np.zeros(3, dtype=float)
        self._gripper_ctrl = self.config.gripper_open_ctrl if gripper_ctrl is None else float(gripper_ctrl)

    def update_action(self, action: Sequence[float]) -> None:
        """Accept a seven-dimensional, already un-normalized OpenVLA action."""

        self._require_ready()
        vector = self._as_vector(action, 7, "action")
        translation = _clip_norm(vector[:3].copy(), self.config.max_translation_step_m)
        self._target = self._bound_workspace(self._target + translation)
        self._rotation_delta = vector[3:6].copy()
        self._gripper_ctrl = (
            self.config.gripper_open_ctrl
            if vector[6] >= self.config.gripper_open_threshold
            else self.config.gripper_closed_ctrl
        )

    def step(self, dt: float) -> CartesianReference:
        """Advance the critically damped reference state by one control step."""

        self._require_ready()
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt!r}")

        omega = 2.0 * np.pi * self.config.natural_frequency_hz
        acceleration = omega**2 * (self._target - self._position)
        acceleration -= 2.0 * self.config.damping_ratio * omega * self._velocity
        acceleration = _clip_norm(acceleration, self.config.max_acceleration_mps2)

        velocity = _clip_norm(self._velocity + acceleration * dt, self.config.max_velocity_mps)
        position = self._bound_workspace(self._position + velocity * dt)

        # Keep the state consistent when a workspace boundary prevents motion.
        if not np.allclose(position, self._position + velocity * dt):
            velocity = (position - self._position) / dt

        self._position = position
        self._velocity = velocity
        return CartesianReference(
            position=position.copy(),
            velocity=velocity.copy(),
            acceleration=acceleration.copy(),
            gripper_ctrl=float(self._gripper_ctrl),
            rotation_delta=self._rotation_delta.copy(),
        )

    def _bound_workspace(self, position: np.ndarray) -> np.ndarray:
        if self.config.workspace_lower is None:
            return position.copy()
        return np.clip(
            position,
            np.asarray(self.config.workspace_lower, dtype=float),
            np.asarray(self.config.workspace_upper, dtype=float),
        )

    @staticmethod
    def _as_vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=float)
        if vector.shape != (size,):
            raise ActionValidationError(f"{name} must have shape ({size},), got {vector.shape}")
        if not np.all(np.isfinite(vector)):
            raise ActionValidationError(f"{name} must contain only finite values")
        return vector

    def _require_ready(self) -> None:
        if not self.is_ready:
            raise RuntimeError("reset() must be called before updating or stepping the adapter")

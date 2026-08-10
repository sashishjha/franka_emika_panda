#!/usr/bin/env python3
"""Collect RGB/language/action demonstrations from the legacy trajectory teacher.

This is an offline data-generation tool.  It is the one place where analytic
trajectory state is allowed to create action labels; the deployed OpenVLA
policy never receives those values.  Each output ``.npz`` contains only policy
inputs/labels plus non-privileged task metadata:

* ``images``: uint8 RGB policy-camera frames;
* ``actions``: [world dx, dy, dz, dRx, dRy, dRz, gripper];
* ``timestamps`` and a single natural-language instruction.

The files are an inspectable staging format.  Convert the reviewed dataset to
the training format required by the selected OpenVLA fine-tuning workflow
before training on RBCCPS.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

from fl_smc_controller import ALL_SHAPES, SHAPE_PROMPTS, run_sim


@dataclass
class EpisodeRecorder:
    """Collect policy-safe observations and teacher action labels for one rollout."""

    instruction: str
    policy_hz: float
    max_translation_step_m: float
    timestamps: List[float] = field(default_factory=list)
    images: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    clipped_actions: int = 0

    def __call__(self, *, t, image, desired_position, desired_velocity,
                 desired_acceleration, actual_position) -> None:
        # The policy sees only image + instruction.  Desired/actual positions
        # are deliberately used here only to generate/validate offline labels.
        del desired_position, desired_acceleration, actual_position
        translation = np.asarray(desired_velocity, dtype=float) / self.policy_hz
        magnitude = float(np.linalg.norm(translation))
        if magnitude > self.max_translation_step_m:
            translation *= self.max_translation_step_m / magnitude
            self.clipped_actions += 1
        action = np.concatenate((translation, np.zeros(3, dtype=float), np.array([1.0])))
        self.timestamps.append(float(t))
        self.images.append(np.asarray(image, dtype=np.uint8).copy())
        self.actions.append(action)

    def save(self, path: Path, metadata: dict) -> None:
        if not self.images:
            raise RuntimeError("Episode contains no samples; check camera/rendering configuration")
        metadata = {
            **metadata,
            "instruction": self.instruction,
            "policy_hz": self.policy_hz,
            "num_clipped_actions": self.clipped_actions,
            "action_spec": {
                "shape": [7],
                "components": ["world_dx_m", "world_dy_m", "world_dz_m", "dRx", "dRy", "dRz", "gripper_open"],
                "rotation_policy": "zero; existing controller is translation-only",
                "gripper_policy": "1.0 means open",
            },
        }
        np.savez_compressed(
            path,
            images=np.stack(self.images),
            actions=np.stack(self.actions),
            timestamps=np.asarray(self.timestamps, dtype=np.float64),
            instruction=np.asarray(self.instruction),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/collected"))
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--policy-hz", type=float, default=5.0)
    parser.add_argument("--episodes-per-shape", type=int, default=1)
    parser.add_argument("--max-translation-step-m", type=float, default=0.03)
    parser.add_argument("--shapes", nargs="+", choices=ALL_SHAPES, default=ALL_SHAPES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration <= 0 or args.policy_hz <= 0 or args.episodes_per_shape <= 0:
        raise SystemExit("duration, policy-hz, and episodes-per-shape must be positive")
    if args.max_translation_step_m <= 0:
        raise SystemExit("max-translation-step-m must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Collecting teacher demonstrations. Policy images exclude debug sites and do not contain MuJoCo state.")
    for shape in args.shapes:
        for episode_index in range(args.episodes_per_shape):
            recorder = EpisodeRecorder(
                instruction=SHAPE_PROMPTS[shape],
                policy_hz=args.policy_hz,
                max_translation_step_m=args.max_translation_step_m,
            )
            print(f"  {shape}: episode {episode_index + 1}/{args.episodes_per_shape}")
            run_sim(
                "SMC",
                shape,
                T_total=args.duration,
                use_viewer=False,
                observation_callback=recorder,
                observation_hz=args.policy_hz,
            )
            path = args.output_dir / f"{shape}_{episode_index:04d}.npz"
            recorder.save(path, {
                "trajectory": shape,
                "teacher_controller": "FL+SMC",
                "scene": "scene_torque.xml",
                "image_source": "policy_camera",
            })
            print(f"    saved {len(recorder.images)} frames to {path}")


if __name__ == "__main__":
    main()

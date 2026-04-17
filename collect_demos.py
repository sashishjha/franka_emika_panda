# collect_demos.py
# Run with:  mjpython collect_demos.py
#            mjpython collect_demos.py --tasks hover_all stack_all   (subset)
#            mjpython collect_demos.py --n 5                          (quick sanity check)
#
# Runs your IK controller for each task, records (image, state, action, instruction)
# at every Nth simulation step, and saves to demos_raw/ in the format that
# build_lerobot_dataset.py expects for SmolVLA fine-tuning.
#
# Designed to run overnight on the IISc cluster unattended.
# On M1 it works too but is slower — cluster recommended.
#
# IMPORTANT: mujoco.mj_step is monkey-patched AFTER importing mujoco but BEFORE
# calling any task functions. Python resolves mujoco.mj_step at call time
# (attribute lookup on the module object), so all calls inside ik_controller_vla
# will use the patched version automatically.

import os
import sys
import json
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from PIL import Image

# ── Parse args BEFORE any heavy imports ───────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n",     type=int, default=50,
                    help="Demos per task (default 50; use 5 for quick sanity check)")
parser.add_argument("--tasks", nargs="+",
                    default=["hover_all","stack_all","stack_by_color",
                             "place_in_bowls","line_up_by_color"],
                    help="Subset of tasks to collect (default: all 5)")
parser.add_argument("--out",   type=str, default="demos_raw",
                    help="Output directory (default: demos_raw)")
parser.add_argument("--every", type=int, default=5,
                    help="Record one frame every N sim steps (default 5 → 50ms at dt=0.01)")
parser.add_argument("--imgsize", type=int, default=224,
                    help="Rendered image size in pixels (default 224)")
parser.add_argument("--camera", type=str, default="cam_top",
                    help="Camera name in scene.xml (default cam_top)")
args = parser.parse_args()

OUTPUT_DIR  = Path(args.out)
N_DEMOS     = args.n
RECORD_EVERY= args.every
IMG_SIZE    = args.imgsize
CAMERA_NAME = args.camera

# ── Natural language instructions (same strings used in SmolVLA training) ─────
TASK_INSTRUCTIONS = {
    "hover_all":        "Hover over all the cubes and inspect each one",
    "stack_all":        "Stack all cubes into a single tall tower",
    "stack_by_color":   "Stack each color's cubes into their own separate tower",
    "place_in_bowls":   "Place each cube into its matching colored bowl",
    "line_up_by_color": "Pick up each color's cubes and line them up neatly",
}

# ── Validate requested tasks ───────────────────────────────────────────────────
VALID_TASKS = list(TASK_INSTRUCTIONS.keys())
for t in args.tasks:
    if t not in VALID_TASKS:
        print(f"[ERROR] Unknown task: '{t}'. Valid: {VALID_TASKS}")
        sys.exit(1)

print(f"\nCollection plan:")
print(f"  Tasks   : {args.tasks}")
print(f"  Demos   : {N_DEMOS} per task ({N_DEMOS * len(args.tasks)} total)")
print(f"  Output  : {OUTPUT_DIR.resolve()}")
print(f"  Camera  : {CAMERA_NAME} @ {IMG_SIZE}×{IMG_SIZE}")
print(f"  Interval: every {RECORD_EVERY} sim steps\n")

# ── Import IK controller AFTER mujoco is imported ─────────────────────────────
# The patch below replaces mujoco.mj_step with our recording wrapper.
# ik_controller_vla calls mujoco.mj_step(model, data) by attribute lookup,
# so it will automatically use the patched version at call time.
import ik_controller_vla as ctrl

# ── Verify camera ──────────────────────────────────────────────────────────────
_cam_id = mujoco.mj_name2id(ctrl.model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
if _cam_id < 0:
    print(f"[ERROR] Camera '{CAMERA_NAME}' not found in scene.xml.")
    print("  Add this inside <worldbody>:")
    print('  <camera name="cam_top" pos="0.45 -0.10 1.30" euler="0 55 85" fovy="65"/>')
    sys.exit(1)

renderer = mujoco.Renderer(ctrl.model, height=IMG_SIZE, width=IMG_SIZE)

# ── Recording state (all mutable, modified by the mj_step patch) ──────────────
_recording    = False
_episode_buf  = []     # list of dicts, one per recorded step
_step_counter = 0
_prev_ee_pos  = None
_prev_ee_quat = None
_current_instr= ""


def _render_frame() -> np.ndarray:
    """Render current sim state from the top camera → (H, W, 3) uint8 RGB."""
    renderer.update_scene(ctrl.data, camera=CAMERA_NAME)
    return renderer.render().copy()   # copy so the buffer isn't overwritten


def _get_ee_state() -> np.ndarray:
    """
    End-effector pose as [x, y, z, qw, qx, qy, qz, grip_normalized].
    grip is ctrl[7] / 255 → [0, 1].  shape (8,).
    """
    pos  = ctrl.data.xpos[ctrl.hand_id].copy()
    quat = ctrl.data.xquat[ctrl.hand_id].copy()   # (qw, qx, qy, qz) MuJoCo convention
    grip = float(ctrl.data.ctrl[7]) / 255.0
    return np.concatenate([pos, quat, [grip]]).astype(np.float32)


def _get_action_delta() -> np.ndarray:
    """
    Delta end-effector action: [dx, dy, dz, dqw, dqx, dqy, dqz, grip].
    SmolVLA's action head predicts these deltas; during training it regresses to them.
    Shape (8,).
    """
    global _prev_ee_pos, _prev_ee_quat
    pos  = ctrl.data.xpos[ctrl.hand_id].copy()
    quat = ctrl.data.xquat[ctrl.hand_id].copy()
    grip = float(ctrl.data.ctrl[7]) / 255.0

    if _prev_ee_pos is None:
        delta_pos  = np.zeros(3)
        delta_quat = np.zeros(4)
    else:
        delta_pos  = pos  - _prev_ee_pos
        delta_quat = quat - _prev_ee_quat   # simple diff; SmolVLA normalises internally

    _prev_ee_pos  = pos.copy()
    _prev_ee_quat = quat.copy()
    return np.concatenate([delta_pos, delta_quat, [grip]]).astype(np.float32)


# ── Monkey-patch mujoco.mj_step ───────────────────────────────────────────────
_orig_mj_step = mujoco.mj_step

def _patched_mj_step(model, data, nstep=1):
    _orig_mj_step(model, data, nstep)
    global _step_counter, _recording
    if not _recording:
        return
    _step_counter += 1
    if _step_counter % RECORD_EVERY != 0:
        return
    _episode_buf.append({
        "image"      : _render_frame(),
        "state"      : _get_ee_state(),
        "action"     : _get_action_delta(),
        "instruction": _current_instr,
        "sim_time"   : float(data.time),
    })

mujoco.mj_step = _patched_mj_step   # all downstream calls now hit this wrapper
print("mujoco.mj_step patched for recording.")


# ── Episode I/O ───────────────────────────────────────────────────────────────
def _save_episode(task_name: str, ep_idx: int) -> Path:
    ep_dir = OUTPUT_DIR / task_name / f"ep_{ep_idx:04d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    images  = [f["image"]    for f in _episode_buf]
    states  = np.array([f["state"]  for f in _episode_buf], dtype=np.float32)
    actions = np.array([f["action"] for f in _episode_buf], dtype=np.float32)
    times   = np.array([f["sim_time"] for f in _episode_buf], dtype=np.float32)

    img_dir = ep_dir / "images"
    img_dir.mkdir(exist_ok=True)
    for i, rgb in enumerate(images):
        Image.fromarray(rgb).save(img_dir / f"{i:06d}.jpg", quality=90)

    np.save(ep_dir / "states.npy",  states)
    np.save(ep_dir / "actions.npy", actions)
    np.save(ep_dir / "times.npy",   times)

    meta = {
        "task"       : task_name,
        "instruction": _current_instr,
        "num_steps"  : len(_episode_buf),
        "ep_index"   : ep_idx,
        "action_min" : float(actions.min()),
        "action_max" : float(actions.max()),
        "action_mean": float(actions.mean()),
    }
    (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return ep_dir


def _episode_is_sane() -> bool:
    """
    Reject episodes that are too short or have near-zero action variance
    (arm barely moved — task probably failed silently).
    """
    if len(_episode_buf) < 20:
        return False
    actions = np.array([f["action"] for f in _episode_buf])
    if actions[:, :3].std() < 1e-5:   # position deltas all near zero
        return False
    return True


# ── Main collection loop ───────────────────────────────────────────────────────
def collect():
    global _recording, _episode_buf, _step_counter
    global _prev_ee_pos, _prev_ee_quat, _current_instr

    OUTPUT_DIR.mkdir(exist_ok=True)
    total_planned = N_DEMOS * len(args.tasks)
    total_saved   = 0
    total_skipped = 0
    wall_start    = time.time()

    with mujoco.viewer.launch_passive(ctrl.model, ctrl.data) as viewer:
        viewer.cam.lookat[:] = [0.45, 0.0, 0.3]
        viewer.cam.distance  = 1.8
        viewer.cam.elevation = -20

        for task_name in args.tasks:
            instruction = TASK_INSTRUCTIONS[task_name]
            task_fn     = ctrl.TASK_MAP[task_name]

            print(f"\n{'━'*62}")
            print(f"Task  : {task_name}")
            print(f"Instr : {instruction}")
            print(f"{'━'*62}")

            ep_saved = 0

            for attempt in range(N_DEMOS * 3):   # allow retries: try up to 3× per target
                if ep_saved >= N_DEMOS:
                    break
                if not viewer.is_running():
                    print("Viewer closed — stopping.")
                    return

                # ── Reset ─────────────────────────────────────────────────────
                ctrl.reset_to_home()
                mujoco.mj_forward(ctrl.model, ctrl.data)   # uses patched step — OK, not recording
                viewer.sync()

                _episode_buf.clear()
                _step_counter  = 0
                _prev_ee_pos   = None
                _prev_ee_quat  = None
                _current_instr = instruction
                _recording     = True

                # ── Execute task ───────────────────────────────────────────────
                t0 = time.time()
                try:
                    task_fn(viewer)
                    success = True
                except Exception as exc:
                    success = False
                    print(f"  [FAIL] ep attempt {attempt}: {exc}")
                finally:
                    _recording = False

                elapsed = time.time() - t0

                # ── Validate & save ────────────────────────────────────────────
                if success and _episode_is_sane():
                    ep_dir = _save_episode(task_name, ep_saved)
                    total_saved += 1
                    ep_saved    += 1
                    pct = total_saved / total_planned * 100
                    wall_elapsed = time.time() - wall_start
                    eta_s = (wall_elapsed / total_saved) * (total_planned - total_saved) if total_saved else 0
                    print(f"  ✓ ep {ep_saved:3d}/{N_DEMOS}"
                          f"  steps={len(_episode_buf)}"
                          f"  t={elapsed:.1f}s"
                          f"  total={pct:.0f}%"
                          f"  ETA={eta_s/60:.1f}min"
                          f"  → {ep_dir.name}")
                else:
                    total_skipped += 1
                    reason = "too short / no motion" if success else "exception"
                    print(f"  ✗ attempt {attempt} skipped ({reason}, {len(_episode_buf)} frames)")

    # ── Summary ────────────────────────────────────────────────────────────────
    wall_total = time.time() - wall_start
    print(f"\n{'━'*62}")
    print(f"Collection complete.")
    print(f"  Saved   : {total_saved} episodes")
    print(f"  Skipped : {total_skipped} attempts")
    print(f"  Wall time: {wall_total/60:.1f} min")
    print(f"  Output  : {OUTPUT_DIR.resolve()}")
    print(f"\nNext step → run:  python build_lerobot_dataset.py")


if __name__ == "__main__":
    collect()

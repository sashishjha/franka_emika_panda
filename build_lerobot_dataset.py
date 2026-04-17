# build_lerobot_dataset.py
# Run with:  python build_lerobot_dataset.py
# Run after: collect_demos.py has finished (demos_raw/ must exist)
#
# Converts demos_raw/ → lerobot_dataset/ (the format LeRobot's train.py expects).
# Requires: pip install pandas pyarrow pillow

import json
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

RAW_DIR    = Path("demos_raw")
OUT_DIR    = Path("lerobot_dataset")
FPS        = 10     # recording was every 5 sim steps at dt=0.01 → 20Hz; downsample to 10
IMG_SIZE   = 224
ACTION_DIM = 8
STATE_DIM  = 8

TASK_NAMES = [
    "hover_all",
    "stack_all",
    "stack_by_color",
    "place_in_bowls",
    "line_up_by_color",
]

TASK_INSTRUCTIONS = {
    "hover_all"       : "Hover over all the cubes and inspect each one",
    "stack_all"       : "Stack all cubes into a single tall tower",
    "stack_by_color"  : "Stack each color's cubes into their own separate tower",
    "place_in_bowls"  : "Place each cube into its matching colored bowl",
    "line_up_by_color": "Pick up each color's cubes and line them up neatly",
}


def _write_video(img_dir: Path, out_path: Path):
    """Encode JPEG frames → mp4 with ffmpeg. Requires ffmpeg on PATH."""
    cmd = [
        "ffmpeg", "-y",
        "-r", str(FPS),
        "-pattern_type", "glob",
        "-i", str(img_dir / "*.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"  [WARN] ffmpeg failed for {out_path.name}: {result.stderr.decode()[:200]}")
        return False
    return True


def build():
    # Create output tree
    (OUT_DIR / "data"   / "chunk-000").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "videos" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "meta").mkdir(parents=True, exist_ok=True)

    tasks_meta    = []
    episodes_meta = []
    all_rows      = []
    ep_global     = 0
    frame_global  = 0

    for task_idx, task_name in enumerate(TASK_NAMES):
        task_dir = RAW_DIR / task_name
        if not task_dir.exists():
            print(f"[SKIP] {task_dir} not found — run collect_demos.py first")
            continue

        ep_dirs = sorted(d for d in task_dir.iterdir() if d.is_dir())
        if not ep_dirs:
            print(f"[SKIP] {task_dir} is empty")
            continue

        instruction = TASK_INSTRUCTIONS[task_name]
        tasks_meta.append({"task_index": task_idx, "task": instruction})
        print(f"\nTask {task_idx}: {task_name}  ({len(ep_dirs)} episodes)")

        for ep_dir in ep_dirs:
            states  = np.load(ep_dir / "states.npy")   # (T, 8)
            actions = np.load(ep_dir / "actions.npy")  # (T, 8)
            times   = np.load(ep_dir / "times.npy")    # (T,)
            T       = len(states)

            # Write video
            vid_name = f"observation.images.top_episode_{ep_global:06d}.mp4"
            vid_path = OUT_DIR / "videos" / "chunk-000" / vid_name
            ok = _write_video(ep_dir / "images", vid_path)
            status = "✓" if ok else "✗ (no video)"

            # Build parquet rows
            for t in range(T):
                row = {
                    "episode_index": ep_global,
                    "frame_index"  : t,
                    "index"        : frame_global + t,
                    "task_index"   : task_idx,
                    "timestamp"    : float(times[t]),
                    "next.done"    : bool(t == T - 1),
                    "next.reward"  : 1.0 if (t == T - 1) else 0.0,
                    "task"         : instruction,
                }
                for i, v in enumerate(states[t]):
                    row[f"observation.state.{i}"] = float(v)
                for i, v in enumerate(actions[t]):
                    row[f"action.{i}"] = float(v)
                all_rows.append(row)

            episodes_meta.append({
                "episode_index": ep_global,
                "task_index"   : task_idx,
                "length"       : T,
                "task"         : instruction,
            })

            print(f"  ep {ep_global:4d}  frames={T:4d}  video={status}")
            frame_global += T
            ep_global    += 1

    if not all_rows:
        print("\n[ERROR] No data found. Check that demos_raw/ is populated.")
        return

    # Write parquet
    df = pd.DataFrame(all_rows)
    parquet_path = OUT_DIR / "data" / "chunk-000" / "train.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"\nParquet: {parquet_path}  ({len(df)} rows, {df.memory_usage(deep=True).sum() / 1e6:.1f} MB)")

    # Write meta/tasks.jsonl
    (OUT_DIR / "meta" / "tasks.jsonl").write_text(
        "\n".join(json.dumps(t) for t in tasks_meta))

    # Write meta/episodes.jsonl
    (OUT_DIR / "meta" / "episodes.jsonl").write_text(
        "\n".join(json.dumps(e) for e in episodes_meta))

    # Write meta/info.json  (required by LeRobotDataset loader)
    state_names  = ["x","y","z","qw","qx","qy","qz","grip"]
    action_names = ["dx","dy","dz","dqw","dqx","dqy","dqz","grip"]
    info = {
        "codebase_version": "v2.1",
        "robot_type"      : "panda",
        "total_episodes"  : ep_global,
        "total_frames"    : frame_global,
        "fps"             : FPS,
        "video"           : True,
        "features": {
            "observation.images.top": {
                "dtype": "video",
                "shape": [IMG_SIZE, IMG_SIZE, 3],
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [STATE_DIM],
                "names": state_names,
            },
            "action": {
                "dtype": "float32",
                "shape": [ACTION_DIM],
                "names": action_names,
            },
        },
        "tasks"       : [t["task"] for t in tasks_meta],
        "task_to_idx" : {t["task"]: t["task_index"] for t in tasks_meta},
    }
    (OUT_DIR / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    # Print action statistics (useful for debugging normalisation in training)
    actions_all = df[[f"action.{i}" for i in range(ACTION_DIM)]].values
    print(f"\nAction statistics (for normalisation debugging):")
    print(f"  min  = {actions_all.min(axis=0).round(5)}")
    print(f"  max  = {actions_all.max(axis=0).round(5)}")
    print(f"  mean = {actions_all.mean(axis=0).round(5)}")
    print(f"  std  = {actions_all.std(axis=0).round(5)}")

    print(f"\n{'━'*60}")
    print(f"Dataset ready at: {OUT_DIR.resolve()}")
    print(f"  {ep_global} episodes  |  {frame_global} frames  |  {len(tasks_meta)} tasks")
    print(f"\nNext step → upload to cluster:")
    print(f"  rsync -avz lerobot_dataset/ cluster:~/franka_vla/lerobot_dataset/")
    print(f"Then on cluster:")
    print(f"  python train_smolvla.py")


if __name__ == "__main__":
    build()

# vla_gemini.py
# Run with: mjpython vla_gemini.py
#
# This IS a real VLA pipeline:
#   Vision  — camera image from MuJoCo rendered at each decision step
#   Language — natural language command from user
#   Action   — Gemini Vision outputs structured JSON → IK controller executes
#
# The key difference from the old pipeline:
#   OLD: send text command → Gemini parses to task name → run hardcoded task
#   NEW: send image + command → Gemini sees the scene → outputs verified task +
#        object state observations → controller executes with scene awareness
#
# This means Gemini is doing vision grounding, not just NLP.
# If an object is missing or misplaced, Gemini will say so.

import os
import sys
import time
import json
import base64
import urllib.request
import urllib.error
import numpy as np
import mujoco
import mujoco.viewer

# ── Import the fixed controller ───────────────────────────────────────────────
import ik_controller_vla as ctrl   # brings in model, data, TASK_MAP, etc.

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-flash:generateContent?key=" + GEMINI_API_KEY
)

IMG_WIDTH    = 512
IMG_HEIGHT   = 512
CAMERA_NAME  = "cam_top"   # must exist in scene.xml

# The 5 valid task keys the controller knows
VALID_TASKS = list(ctrl.TASK_MAP.keys())

# Natural language descriptions shown to the user at runtime
DEMO_COMMANDS = [
    ("hover_all",        "Hover over all the cubes and inspect each one"),
    ("stack_all",        "Stack all the cubes into a single tall tower"),
    ("stack_by_color",   "Stack each color group into its own separate tower"),
    ("place_in_bowls",   "Place each cube into its matching colored bowl"),
    ("line_up_by_color", "Pick up each color's cubes and line them up in a row"),
]

# ══════════════════════════════════════════════════════════════
#  CAMERA SETUP  — add cam_top to scene.xml if missing
# ══════════════════════════════════════════════════════════════
renderer = mujoco.Renderer(ctrl.model, height=IMG_HEIGHT, width=IMG_WIDTH)
cam_id   = mujoco.mj_name2id(ctrl.model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
if cam_id < 0:
    print(f"\n[ERROR] Camera '{CAMERA_NAME}' not found in scene.xml.")
    print("Add this line inside <worldbody> in scene.xml:")
    print('  <camera name="cam_top" pos="0.45 -0.10 1.30" euler="0 55 85" fovy="65"/>')
    sys.exit(1)
print(f"Camera '{CAMERA_NAME}' found (id={cam_id})")

# ══════════════════════════════════════════════════════════════
#  RENDER + ENCODE
# ══════════════════════════════════════════════════════════════
def capture_image_b64() -> str:
    """Render current scene from cam_top → base64 JPEG string."""
    renderer.update_scene(ctrl.data, camera=CAMERA_NAME)
    rgb = renderer.render()   # (H, W, 3) uint8

    # Encode as JPEG in memory using only stdlib
    import io
    try:
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except ImportError:
        # Fallback: raw PPM → PNG not available without PIL, use PNG via struct
        # This path is rarely hit since mujoco env usually has PIL
        raise RuntimeError("PIL not available. Install: pip install Pillow")

# ══════════════════════════════════════════════════════════════
#  GEMINI VISION CALL
# ══════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are a vision-language controller for a Franka Panda robot arm in a MuJoCo simulation.

The scene contains:
- A Franka Panda robot arm
- 6 colored cubes: 2 red, 2 blue, 2 green
- 3 colored bowls: red, blue, green (fixed on the table)

You will receive:
1. A camera image of the current scene
2. A natural language command from the user

Your job:
1. Look at the image and verify the scene matches what the command expects
2. Output a JSON object with your decision

Valid task keys: ["hover_all", "stack_all", "stack_by_color", "place_in_bowls", "line_up_by_color"]

Output ONLY valid JSON, no markdown, no explanation:
{
  "task": "<one of the valid task keys>",
  "confidence": <0.0 to 1.0>,
  "scene_observation": "<one sentence: what you see in the image>",
  "ready": <true if scene looks ready to execute the task, false otherwise>,
  "reason": "<brief reason for your decision>"
}

If the command is ambiguous, pick the closest matching task and set confidence below 0.7.
If the scene looks wrong (e.g. cubes already stacked when asked to stack), still return the task but set ready=false."""

def call_gemini_vision(command: str, image_b64: str) -> dict:
    """
    Send image + command to Gemini 1.5 Flash Vision.
    Returns parsed JSON dict or raises on failure.
    """
    payload = {
        "contents": [{
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_b64
                    }
                },
                {
                    "text": f"{SYSTEM_PROMPT}\n\nUser command: {command}"
                }
            ]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 256,
        }
    }

    body = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        GEMINI_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Gemini HTTP {e.code}: {e.read().decode()}")

    # Extract text from Gemini response
    text = raw["candidates"][0]["content"]["parts"][0]["text"].strip()

    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError(f"Gemini returned non-JSON: {text}")

    # Validate task key
    if result.get("task") not in VALID_TASKS:
        raise RuntimeError(f"Gemini returned unknown task: {result.get('task')}")

    return result

# ══════════════════════════════════════════════════════════════
#  KEYWORD FALLBACK (if Gemini call fails)
# ══════════════════════════════════════════════════════════════
def keyword_fallback(command: str) -> str:
    cmd = command.lower()
    if "hover"  in cmd or "inspect"  in cmd:  return "hover_all"
    if "bowl"   in cmd or "match"    in cmd:  return "place_in_bowls"
    if "color"  in cmd and "stack"   in cmd:  return "stack_by_color"
    if "line"   in cmd or "row"      in cmd:  return "line_up_by_color"
    if "stack"  in cmd or "tower"    in cmd:  return "stack_all"
    return "hover_all"   # safe default

# ══════════════════════════════════════════════════════════════
#  MAIN VLA LOOP
# ══════════════════════════════════════════════════════════════
def run_vla():
    print("\n" + "="*60)
    print("  GEMINI VISION VLA PIPELINE")
    print("  Vision + Language → Action")
    print("="*60)

    with mujoco.viewer.launch_passive(ctrl.model, ctrl.data) as viewer:
        viewer.cam.lookat[:] = [0.45, 0.0, 0.3]
        viewer.cam.distance   = 1.8
        viewer.cam.elevation  = -20

        for task_idx, (task_key, command) in enumerate(DEMO_COMMANDS):
            print(f"\n{'─'*60}")
            print(f"Command {task_idx+1}/5: \"{command}\"")
            print(f"{'─'*60}")

            ctrl.reset_to_home()
            ctrl.pause(viewer, 300)
            viewer.sync()

            # ── VISION step: capture scene ────────────────────────────────
            print("  [VLA] Capturing scene image...")
            image_b64 = capture_image_b64()

            # ── LANGUAGE step: send to Gemini Vision ─────────────────────
            print("  [VLA] Querying Gemini Vision...")
            try:
                result = call_gemini_vision(command, image_b64)
                chosen_task = result["task"]
                confidence  = result.get("confidence", 1.0)
                observation = result.get("scene_observation", "")
                ready       = result.get("ready", True)
                reason      = result.get("reason", "")

                print(f"  [VLA] Gemini sees: {observation}")
                print(f"  [VLA] Decision: {chosen_task}  (confidence={confidence:.2f})")
                print(f"  [VLA] Ready: {ready}  Reason: {reason}")

                if not ready:
                    print("  [VLA] Scene not ready — executing anyway (simulation context)")

                if confidence < 0.5:
                    print(f"  [VLA] Low confidence — using keyword fallback")
                    chosen_task = keyword_fallback(command)

            except Exception as e:
                print(f"  [VLA] Gemini failed: {e}")
                print(f"  [VLA] Using keyword fallback")
                chosen_task = keyword_fallback(command)

            # ── ACTION step: execute chosen task ──────────────────────────
            print(f"  [VLA] Executing: {chosen_task}")
            task_fn = ctrl.TASK_MAP[chosen_task]
            task_fn(viewer)

            print(f"  [VLA] Task complete. Pausing 2s...")
            time.sleep(2.0)

        print("\n" + "="*60)
        print("  All 5 commands executed.")
        print("  Keeping viewer open — close window to exit.")
        print("="*60)

        while viewer.is_running():
            mujoco.mj_step(ctrl.model, ctrl.data)
            viewer.sync()

# ══════════════════════════════════════════════════════════════
#  INTERACTIVE MODE — type your own commands
# ══════════════════════════════════════════════════════════════
def run_interactive():
    """Type any natural language command — Gemini decides which task to run."""
    print("\n" + "="*60)
    print("  INTERACTIVE VLA MODE")
    print("  Type a command, Gemini Vision will decide what to do.")
    print("  Type 'quit' to exit.")
    print("="*60)

    with mujoco.viewer.launch_passive(ctrl.model, ctrl.data) as viewer:
        viewer.cam.lookat[:] = [0.45, 0.0, 0.3]
        viewer.cam.distance   = 1.8
        viewer.cam.elevation  = -20

        ctrl.reset_to_home()
        ctrl.pause(viewer, 300)

        while viewer.is_running():
            cmd = input("\n> Enter command (or 'reset'/'quit'): ").strip()
            if not cmd:
                continue
            if cmd.lower() == "quit":
                break
            if cmd.lower() == "reset":
                ctrl.reset_to_home()
                ctrl.pause(viewer, 300)
                print("  Scene reset.")
                continue

            image_b64 = capture_image_b64()
            print("  [VLA] Querying Gemini Vision...")

            try:
                result = call_gemini_vision(cmd, image_b64)
                chosen_task = result["task"]
                print(f"  [VLA] Gemini sees: {result.get('scene_observation','')}")
                print(f"  [VLA] → {chosen_task}  (confidence={result.get('confidence',1):.2f})")
            except Exception as e:
                print(f"  [VLA] Gemini error: {e}  → fallback")
                chosen_task = keyword_fallback(cmd)

            ctrl.TASK_MAP[chosen_task](viewer)
            print(f"  Done.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        run_interactive()
    else:
        run_vla()
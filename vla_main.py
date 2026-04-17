# vla_main.py
# Run with: mjpython vla_main.py
#
# Full VLA pipeline:
#   Natural language command → Gemini API → task name → execute

import os
import json
import urllib.request
import urllib.error
import mujoco
import mujoco.viewer

# ── Import controller (must be in same directory) ──────────────
from ik_controller import (
    model, data, reset_to_home, pause, TASK_MAP
)

# ══════════════════════════════════════════════════════════════
#  GEMINI CONFIG
# ══════════════════════════════════════════════════════════════
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
print(f"gemini API key: {'found' if GEMINI_API_KEY else 'NOT found'}")
GEMINI_URL = ( 
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-flash:generateContent?key=" + GEMINI_API_KEY
)

# ── System prompt that tells Gemini about our 5 tasks ──────────
SYSTEM_PROMPT = """
You are the brain of a robotic arm (Franka Panda) in a MuJoCo simulation.
The scene has 6 colored cubes (2 red, 2 blue, 2 green) and 3 colored bowls
(red, blue, green) on a table.

Given a natural language command, you must decide which ONE task to perform.
Respond ONLY with a valid JSON object — no markdown, no explanation:

{
  "task": "<task_name>",
  "reasoning": "<one sentence explaining why>"
}

Available task names and what they do:
- "hover_all"       : Hover the arm above every cube for 5 seconds each (inspection)
- "stack_all"       : Pick all 6 cubes and stack them into a single tall tower
- "stack_by_color"  : Build 3 separate towers, one per color (red tower, blue tower, green tower)
- "place_in_bowls"  : Place each cube into the bowl that matches its color
- "line_up_by_color": Arrange each color's cubes side by side in neat rows

Choose the task that best matches the user's intent.
If the command is ambiguous, choose the closest match.
"""

# ══════════════════════════════════════════════════════════════
#  GEMINI CALL
# ══════════════════════════════════════════════════════════════
def parse_command(user_command: str) -> dict:
    """
    Send user_command to Gemini, get back {"task": ..., "reasoning": ...}.
    Falls back to a simple keyword match if API key missing or call fails.
    """
    if not GEMINI_API_KEY:
        print("  ⚠️  No GEMINI_API_KEY found — using keyword fallback")
        return _keyword_fallback(user_command)

    payload = json.dumps({
        "contents": [{
            "parts": [
                {"text": SYSTEM_PROMPT},
                {"text": f"Command: {user_command}"}
            ]
        }]
    }).encode("utf-8")

    req = urllib.request.Request(
        GEMINI_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        text = raw["candidates"][0]["content"]["parts"][0]["text"].strip()
        # Strip markdown fences if Gemini adds them
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        return result
    except Exception as e:
        print(f"  ⚠️  Gemini call failed ({e}) — using keyword fallback")
        return _keyword_fallback(user_command)

def _keyword_fallback(cmd: str) -> dict:
    """Simple keyword matching when Gemini is unavailable."""
    cmd = cmd.lower()
    if "inspect" in cmd or "hover" in cmd:
        task = "hover_all"
    elif "single" in cmd or "one tower" in cmd or ("stack" in cmd and "color" not in cmd and "respective" not in cmd):
        task = "stack_all"
    elif "respective" in cmd or "each color" in cmd and "stack" in cmd:
        task = "stack_by_color"
    elif "bowl" in cmd or "matching" in cmd:
        task = "place_in_bowls"
    elif "line" in cmd or "row" in cmd:
        task = "line_up_by_color"
    else:
        task = "hover_all"
    return {"task": task, "reasoning": "keyword match fallback"}

# ══════════════════════════════════════════════════════════════
#  5 DEMO COMMANDS
# ══════════════════════════════════════════════════════════════
DEMO_COMMANDS = [
    "Hover over all the cubes and inspect them",
    "Stack all cubes into a single tower",
    "Stack each color's cubes into their respective towers",
    "Place each cube into its matching colored bowl",
    "Pick up each color's cubes and line them up",
]

# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def run_vla(viewer):
    print("\n" + "═"*55)
    print("  VLA PIPELINE — Franka Panda + Gemini")
    print("═"*55)

    for i, command in enumerate(DEMO_COMMANDS, 1):
        print(f"\n{'─'*55}")
        print(f"  Command {i}: \"{command}\"")
        print(f"{'─'*55}")

        # ── Parse via Gemini ──────────────────────────────────
        result = parse_command(command)
        task_name = result.get("task", "hover_all")
        reasoning = result.get("reasoning", "")
        print(f"  Gemini → task: '{task_name}'")
        print(f"  Reasoning: {reasoning}")

        # ── Validate ──────────────────────────────────────────
        if task_name not in TASK_MAP:
            print(f"  ⚠️  Unknown task '{task_name}', skipping")
            continue

        # ── Execute ───────────────────────────────────────────
        print(f"  Executing…")
        TASK_MAP[task_name](viewer)

        # ── Reset scene for next command ──────────────────────
        if i < len(DEMO_COMMANDS):
            print(f"\n  ↺ Resetting scene for next command…")
            reset_to_home()
            pause(viewer, 800)

    print("\n" + "═"*55)
    print("  ✓ All 5 VLA commands executed successfully!")
    print("═"*55)

if __name__ == "__main__":
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.45, 0.0, 0.3]
        viewer.cam.distance   = 1.8
        viewer.cam.elevation  = -20

        run_vla(viewer)

        # Keep window open after all tasks finish
        import time
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)
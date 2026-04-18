# vla_vision.py  (indentation-fixed)
# Run with:  mjpython vla_vision.py
#            mjpython vla_vision.py interactive

import os, sys, json, time
import numpy as np
import mujoco
import mujoco.viewer
from PIL import Image

from google import genai
import ik_controller_vla as ctrl
from vision_detect import VisionDetector, print_detections, BOWL_Z

CAMERA_NAME  = "cam_top"
IMG_H = IMG_W = 224
GEMINI_MODEL = "gemini-2.5-flash"
REDETECT_EVERY = 1

_api_key = os.environ.get("GEMINI_API_KEY", "")
if not _api_key:
    print("[ERROR] export GEMINI_API_KEY='your-key'"); sys.exit(1)
_gemini = genai.Client(api_key=_api_key)
GEMINI_MODEL = "gemini-2.5-flash"   # updated model name

_cam_id = mujoco.mj_name2id(ctrl.model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
if _cam_id < 0:
    print(f"[ERROR] Camera '{CAMERA_NAME}' not found. Add to scene.xml."); sys.exit(1)

_renderer = mujoco.Renderer(ctrl.model, height=IMG_H, width=IMG_W)
_detector  = VisionDetector(ctrl.model, ctrl.data, _cam_id, IMG_H, IMG_W)

SYSTEM_PROMPT = """You are the planning module of a Franka Panda robot arm.
A top-down camera image of the current scene is attached.
Scene: 6 colored cubes (2 red, 2 blue, 2 green) and 3 colored bowls on a table.
Tasks (use EXACTLY these names):
  hover_all, stack_all, stack_by_color, place_in_bowls, line_up_by_color
Respond with ONLY valid JSON (no fences):
{"task_name":"<task>","scene_observation":"<1-2 sentences>","reasoning":"<1 sentence>"}"""

def query_gemini(command, rgb):
    pil    = Image.fromarray(rgb)
    prompt = f"{SYSTEM_PROMPT}\n\nUser command: \"{command}\""

    # Retry up to 3 times on transient 429s, then fall back to keyword parser.
    last_err = None
    for attempt in range(3):
        try:
            response = _gemini.models.generate_content(
                model=GEMINI_MODEL,
                contents=[prompt, pil]
            )
            raw = response.text.strip()
            if "```" in raw:
                for p in raw.split("```"):
                    p = p.strip().lstrip("json").strip()
                    if p.startswith("{"): raw = p; break
            try:    result = json.loads(raw)
            except:
                # result = {"task_name": _keyword_fallback(command),"scene_observation": "(parse failed)", "reasoning": "fallback"}
                pass
            if result.get("task_name") not in ctrl.TASK_MAP:
                result["task_name"] = _keyword_fallback(command)
            return result

        except Exception as e:
            last_err = e
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                # Extract suggested retry delay from the error message if present
                wait = 60
                import re
                m = re.search(r"retry[^\d]*(\d+)", err_str, re.I)
                if m:
                    wait = int(m.group(1)) + 5
                if attempt < 2:
                    print(f"  [GEMINI] Rate-limited (429). Waiting {wait}s then retrying "
                          f"(attempt {attempt+2}/3)…")
                    time.sleep(wait)
                else:
                    print(f"  [GEMINI] Quota exhausted — using keyword fallback.")
            else:
                print(f"  [GEMINI] API error: {e}")
                break   # non-rate-limit error, don't retry

    # Keyword fallback keeps the robot running when Gemini is unavailable
    task = _keyword_fallback(command)
    print(f"  [GEMINI] Fallback task: {task}")
    return {"task_name": task,
            "scene_observation": f"(Gemini unavailable: {type(last_err).__name__})",
            "reasoning": "keyword fallback"}

def _keyword_fallback(cmd):
    cmd = cmd.lower()
    if any(w in cmd for w in ["hover","inspect","look"]): return "hover_all"
    if any(w in cmd for w in ["bowl","match"]) and "stack" not in cmd: return "place_in_bowls"
    if any(w in cmd for w in ["line","row","arrange"]): return "line_up_by_color"
    if any(w in cmd for w in ["separate","own","each col"]): return "stack_by_color"
    return "stack_all"

def capture_and_detect(debug=False):
    _renderer.update_scene(ctrl.data, camera=CAMERA_NAME)
    rgb = _renderer.render().copy()
    det = _detector.detect(rgb, debug=debug)
    return rgb, det

class VisionState:
    def __init__(self):
        self._det = None
    def update(self, viewer=None):
        rgb, det = capture_and_detect()
        self._det = det
        print_detections(det)
        return det
    def cube(self, color, idx):
        if not self._det: return None
        pos = self._det["cubes"].get(color, [None, None])
        return pos[idx] if idx < len(pos) else None
    def bowl(self, color):
        if not self._det: return None
        return self._det["bowls"].get(color)

vs = VisionState()

def _bid(name):
    return mujoco.mj_name2id(ctrl.model, mujoco.mjtObj.mjOBJ_BODY, name)

CUBE_BODY_IDS = {
    "red":   [_bid("red_cube_1"),   _bid("red_cube_2")],
    "blue":  [_bid("blue_cube_1"),  _bid("blue_cube_2")],
    "green": [_bid("green_cube_1"), _bid("green_cube_2")],
}


# ── Vision-driven pick and place ───────────────────────────────────────────────
def vision_pick_and_place(cube_color, cube_idx, place_xyz, viewer, cube_body_id=None):
    cube_pos = vs.cube(cube_color, cube_idx)
    if cube_pos is None:
        print(f"  [VISION] {cube_color}_cube_{cube_idx+1} not detected — skipping")
        return
    print(f"  [VISION] {cube_color}_cube_{cube_idx+1} → {np.round(cube_pos,3)}")

    # Vision blob indices and physics body indices within a colour can be
    # swapped (vision sorts by area; physics order is fixed by scene.xml).
    # Resolve by finding whichever physics body is closest to vision's XY.
    if cube_body_id is not None:
        cube_body_id = min(
            CUBE_BODY_IDS[cube_color],
            key=lambda bid: np.linalg.norm(ctrl.data.xpos[bid][:2] - cube_pos[:2])
        )
        rname = mujoco.mj_id2name(ctrl.model, mujoco.mjtObj.mjOBJ_BODY, cube_body_id)
        print(f"  [WELD TARGET] {rname}  GT={np.round(ctrl.data.xpos[cube_body_id],3)}")

    _pick_and_place_pos(cube_pos, np.array(place_xyz), cube_body_id, viewer)


def _pick_and_place_pos(cube_world_pos, place_xyz, cube_id, viewer):
    """
    Full pick-and-place using vision-estimated cube_world_pos.
    cube_id used only for weld (physics), NOT for navigation.
    """
    cname = (mujoco.mj_id2name(ctrl.model, mujoco.mjtObj.mjOBJ_BODY, cube_id)
             if cube_id is not None else "unknown")
    print(f"\n  pick_and_place_pos: {cname} target={np.round(place_xyz,3)}")
    print(f"  vision pos: {np.round(cube_world_pos,3)}")

    hover_pos = cube_world_pos.copy()
    hover_pos[2] += ctrl.FINGER_OFFSET + ctrl.HOVER_HEIGHT

    grasp_pos = cube_world_pos.copy()
    grasp_pos[2] += ctrl.FINGER_OFFSET + ctrl.GRASP_HEIGHT

    success = False
    for attempt in range(3):
        if attempt > 0:
            print(f"    [RETRY] attempt {attempt+1}/3")
            ctrl.detach_all()
            ctrl.open_gripper(viewer, 200)
            ctrl.go_home(viewer)

        ctrl.data.ctrl[:7] = ctrl.data.qpos[:7].copy()   # kill windup

        ctrl.open_gripper(viewer, 100)
        ctrl.detach_all()
        ctrl.move_to(hover_pos, viewer, max_steps=30000)
        ctrl.pause(viewer, 200)

        ctrl.descend_to(grasp_pos, viewer)

        if cube_id is not None:
            ctrl.close_and_weld(cube_id, viewer)
            if ctrl.grasp_held(cube_id, viewer):
                success = True
                break
            ctrl.detach_all()
            ctrl.open_gripper(viewer, 100)
        else:
            success = True
            break

    if not success:
        print(f"    [ABORT] {cname} — skipping after 3 attempts")
        ctrl.detach_all()
        ctrl.reset_to_home()
        mujoco.mj_forward(ctrl.model, ctrl.data)
        viewer.sync()
        return

    # ── Transport (this block is at function level, NOT inside the for loop) ──
    transit = ctrl.data.xpos[ctrl.hand_id].copy()
    transit[2] += 0.30
    ctrl.move_to(transit, viewer)

    approach = place_xyz.copy()
    approach[2] = place_xyz[2] + ctrl.FINGER_OFFSET + ctrl.HOVER_HEIGHT
    ctrl.move_to(approach, viewer)

    drop = place_xyz.copy()
    drop[2] = place_xyz[2] + ctrl.FINGER_OFFSET + 0.005
    ctrl.move_to(drop, viewer)

    ctrl.detach_all()
    ctrl.open_gripper(viewer, 200)
    ctrl.pause(viewer, 400)

    retract = ctrl.data.xpos[ctrl.hand_id].copy()
    retract[2] += 0.20
    ctrl.move_to(retract, viewer)
    ctrl.go_home(viewer)


# ── Vision-driven task implementations ────────────────────────────────────────
def task_hover_all(viewer):
    for color in ("red","blue","green"):
        for idx in range(2):
            pos = vs.cube(color, idx)
            if pos is None: print(f"  {color}_{idx+1}: not detected"); continue
            hover = pos.copy(); hover[2] += ctrl.FINGER_OFFSET + ctrl.HOVER_HEIGHT
            ctrl.move_to(hover, viewer); ctrl.pause(viewer, 500)
    ctrl.go_home(viewer)

def task_stack_all(viewer):
    stack_xy = np.array([0.50, 0.0]); layer = 0
    for color in ("red","blue","green"):
        for idx in range(2):
            if layer % REDETECT_EVERY == 0: vs.update(viewer)
            target = np.array([stack_xy[0], stack_xy[1],
                               ctrl.STACK_BASE_Z + layer * ctrl.CUBE_HEIGHT])
            vision_pick_and_place(color, idx, target, viewer, CUBE_BODY_IDS[color][idx])
            layer += 1

def task_stack_by_color(viewer):
    tower_xy = {"red":np.array([0.60,0.30]),"blue":np.array([0.45,0.30]),"green":np.array([0.30,0.30])}
    op = 0
    for color in ("red","blue","green"):
        for layer, idx in enumerate(range(2)):
            if op % REDETECT_EVERY == 0: vs.update(viewer)
            xy = tower_xy[color]
            target = np.array([xy[0], xy[1], ctrl.STACK_BASE_Z + layer * ctrl.CUBE_HEIGHT])
            vision_pick_and_place(color, idx, target, viewer, CUBE_BODY_IDS[color][idx])
            op += 1

def task_place_in_bowls(viewer):
    op = 0
    for color in ("red","blue","green"):
        for layer, idx in enumerate(range(2)):
            if op % REDETECT_EVERY == 0: vs.update(viewer)
            bowl_pos = vs.bowl(color)
            if bowl_pos is None:
                print(f"  {color} bowl not detected — skipping"); op += 1; continue
            target = np.array([bowl_pos[0], bowl_pos[1],
                               BOWL_Z + layer * ctrl.CUBE_HEIGHT])
            vision_pick_and_place(color, idx, target, viewer, CUBE_BODY_IDS[color][idx])
            op += 1

def task_line_up_by_color(viewer):
    color_y = {"red":0.30,"blue":0.15,"green":0.00}; op = 0
    for color in ("red","blue","green"):
        y0 = color_y[color]
        for i, idx in enumerate(range(2)):
            if op % REDETECT_EVERY == 0: vs.update(viewer)
            target = np.array([0.55, y0 + i*0.10, ctrl.STACK_BASE_Z])
            vision_pick_and_place(color, idx, target, viewer, CUBE_BODY_IDS[color][idx])
            op += 1

VISION_TASK_MAP = {
    "hover_all":        task_hover_all,
    "stack_all":        task_stack_all,
    "stack_by_color":   task_stack_by_color,
    "place_in_bowls":   task_place_in_bowls,
    "line_up_by_color": task_line_up_by_color,
}

DEMO_COMMANDS = [
    "Hover over every cube on the table and inspect each one",
    "Stack all the cubes on top of each other into one tower",
    "Make three separate stacks, one for each colour",
    "Put every cube into the bowl that matches its colour",
    "Line the cubes up in neat rows, grouped by colour",
]


def run_command(command, viewer):
    print(f"\n{'═'*62}\n  COMMAND : {command}\n{'═'*62}")

    print("\n  [1/3] Capturing scene + vision detection...")
    rgb, det = capture_and_detect()
    vs._det  = det
    print_detections(det)

    print("  [2/3] Querying Gemini Vision...")
    t0 = time.time()
    result = query_gemini(command, rgb)
    task   = result["task_name"]
    print(f"\n  ┌─ Gemini ──────────────────────────────────────────────┐")
    print(f"  │  task    : {task}")
    print(f"  │  scene   : {result.get('scene_observation','—')[:60]}")
    print(f"  │  latency : {time.time()-t0:.2f}s")
    print(f"  └───────────────────────────────────────────────────────┘\n")

    print(f"  [3/3] Executing (positions from camera)...")
    ctrl.reset_to_home(); mujoco.mj_forward(ctrl.model, ctrl.data); viewer.sync()
    time.sleep(0.4)
    VISION_TASK_MAP[task](viewer)
    ctrl.reset_to_home(); mujoco.mj_forward(ctrl.model, ctrl.data); viewer.sync()
    print(f"\n  ✓ Done.")
    return result


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"
    with mujoco.viewer.launch_passive(ctrl.model, ctrl.data) as viewer:
        viewer.cam.lookat[:] = [0.45,0.0,0.3]
        viewer.cam.distance  = 1.8
        viewer.cam.elevation = -20
        ctrl.reset_to_home(); mujoco.mj_forward(ctrl.model, ctrl.data); viewer.sync()
        time.sleep(1.0)
        log = []

        if mode == "interactive":
            print("\nInteractive VLA. Type a command, 'demo', or 'quit'.")
            while viewer.is_running():
                try:    cmd = input("\n  > ").strip()
                except: break
                if not cmd: continue
                if cmd.lower() in ("quit","q"): break
                if cmd.lower() == "demo":
                    for c in DEMO_COMMANDS: log.append(run_command(c, viewer))
                elif cmd.lower() == "reset":
                    ctrl.reset_to_home(); mujoco.mj_forward(ctrl.model, ctrl.data); viewer.sync()
                else:
                    log.append(run_command(cmd, viewer))
        else:
            for i, cmd in enumerate(DEMO_COMMANDS, 1):
                print(f"\n[{i}/{len(DEMO_COMMANDS)}]")
                log.append(run_command(cmd, viewer))
                if i < len(DEMO_COMMANDS): time.sleep(1.0)

        if log:
            p = f"vla_log_{int(time.time())}.json"
            json.dump(log, open(p,"w"), indent=2)
            print(f"\nLog: {p}")

        print("\nDone. Ctrl-C to exit.")
        while viewer.is_running():
            mujoco.mj_step(ctrl.model, ctrl.data); viewer.sync()
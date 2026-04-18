import os, sys, json, time, re
import numpy as np
import mujoco
import mujoco.viewer
from PIL import Image

from google import genai
import ik_controller_vla as ctrl
from vision_detect import VisionDetector, print_detections, BOWL_Z

# ── CONFIG ────────────────────────────────────────────────────────────────────
CAMERA_NAME    = "cam_top"       
IMG_H          = 224
IMG_W          = 224
GEMINI_MODEL   = "gemini-2.5-flash"
REDETECT_EVERY = 1              

# ── Gemini client ─────────────────────────────────────────────────────────────
_api_key = os.environ.get("GEMINI_API_KEY", "")
if not _api_key:
    print("[ERROR] export GEMINI_API_KEY='your-key'"); sys.exit(1)
_gemini = genai.Client(api_key=_api_key)

# ── MuJoCo camera + renderer + detector ──────────────────────────────────────
_cam_id = mujoco.mj_name2id(ctrl.model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
if _cam_id < 0:
    print(f"[ERROR] Camera '{CAMERA_NAME}' not found in scene.xml."); sys.exit(1)

_renderer = mujoco.Renderer(ctrl.model, height=IMG_H, width=IMG_W)
_detector  = VisionDetector(ctrl.model, ctrl.data, _cam_id, IMG_H, IMG_W)
_frame_count = 0

COLORS = ("red", "blue", "green")

def _bid(name):
    return mujoco.mj_name2id(ctrl.model, mujoco.mjtObj.mjOBJ_BODY, name)

# Body IDs used ONLY for weld (physics attachment), never for navigation
CUBE_BODY_IDS = {
    c: [_bid(f"{c}_cube_1"), _bid(f"{c}_cube_2")] for c in COLORS
}

# ── Startup banner ────────────────────────────────────────────────────────────
def print_config():
    fovy = ctrl.model.cam_fovy[_cam_id]
    f    = (IMG_H / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
    mm_per_px = (1.0 / f) * 1000   
    print(f"║  Camera        : {CAMERA_NAME:<40}║")
    print(f"║  Image size    : {IMG_W}x{IMG_H} px{' '*35}║")
    print(f"║  Camera fovy   : {fovy:.1f} deg   focal_len={f:.1f}px{' '*16}║")
    print(f"║  px->world @1m : ~{mm_per_px:.1f} mm/px{' '*32}║")
    print(f"║  Gemini model  : {GEMINI_MODEL:<40}║")
    print(f"║  Re-detect     : every {REDETECT_EVERY} pick(s){' '*32}║")
   


# ── Capture + detect ──────────────────────────────────────────────────────────
def capture_and_detect(debug=False):
    global _frame_count
    _frame_count += 1
    t0 = time.time()
    _renderer.update_scene(ctrl.data, camera=CAMERA_NAME)
    rgb = _renderer.render().copy()
    det = _detector.detect(rgb, debug=debug)
    elapsed = time.time() - t0

    counts  = det.get("detected_counts", {})
    n_cubes = sum(v[0] for v in counts.values())
    n_bowls = sum(v[1] for v in counts.values())
    print(f"  [CAM:{CAMERA_NAME}] frame={_frame_count}  "
          f"res={IMG_W}x{IMG_H}  "
          f"detect={elapsed*1000:.0f}ms  "
          f"cubes={n_cubes}/6  bowls={n_bowls}/3")
    return rgb, det


# ── Vision state ──────────────────────────────────────────────────────────────
class VisionState:
    def __init__(self):
        self._det   = None
        self._count = 0

    def update(self, viewer=None):
        self._count += 1
        print(f"\n  -- Vision Update #{self._count} --")
        rgb, det = capture_and_detect()
        self._det = det
        print_detections(det)
        return det

    def cube(self, color, idx):
        if not self._det: return None
        pts = self._det["cubes"].get(color, [])
        return pts[idx] if idx < len(pts) else None

    def bowl(self, color):
        if not self._det: return None
        return self._det["bowls"].get(color)

    def all_cube_positions(self):
        """Flat list of all detected cube world positions."""
        pts = []
        if not self._det: return pts
        for c in COLORS:
            for p in self._det["cubes"].get(c, []):
                if p is not None: pts.append(p)
        return pts

    def color_cube_positions(self, color):
        """Detected cube positions for one color."""
        if not self._det: return []
        return [p for p in self._det["cubes"].get(color, []) if p is not None]

    def scene_centroid(self):
        """XY centroid of all detected cubes — dynamic stack origin."""
        pts = self.all_cube_positions()
        if not pts: return np.array([0.5, 0.0])
        return np.array(pts)[:, :2].mean(axis=0)

    def color_centroid(self, color):
        """XY centroid of detected cubes for one color."""
        pts = self.color_cube_positions(color)
        if not pts: return None
        return np.array(pts)[:, :2].mean(axis=0)

    def scene_bounds(self):
        """Returns (x_min, x_max, y_min, y_max) of all detected cubes."""
        pts = self.all_cube_positions()
        if not pts: return 0.3, 0.7, -0.3, 0.3
        arr = np.array(pts)
        return arr[:,0].min(), arr[:,0].max(), arr[:,1].min(), arr[:,1].max()

vs = VisionState()


# ── Gemini: task classification only ─────────────────────────────────────────
TASK_NAMES = list(ctrl.TASK_MAP.keys())
SYSTEM_PROMPT = f"""You are the planning module of a Franka Panda robot arm.
A top-down camera image of the current scene is attached.
Scene: colored cubes and bowls on a table.
Choose the task that best matches the user command from this exact list:
  {', '.join(TASK_NAMES)}
Respond with ONLY valid JSON, no markdown fences:
{{"task_name":"<task>","scene_observation":"<1-2 sentences describing what you see>","reasoning":"<1 sentence>"}}"""

def query_gemini(command, rgb):
    pil    = Image.fromarray(rgb)
    prompt = f"{SYSTEM_PROMPT}\n\nUser command: \"{command}\""
    last_err = None

    for attempt in range(3):
        try:
            resp = _gemini.models.generate_content(
                model=GEMINI_MODEL,
                contents=[prompt, pil]
            )
            raw = resp.text.strip()
            if "```" in raw:
                for part in raw.split("```"):
                    part = part.strip().lstrip("json").strip()
                    if part.startswith("{"): raw = part; break
            try:    result = json.loads(raw)
            except: result = {}
            if result.get("task_name") not in TASK_NAMES:
                print(f"  [GEMINI] Invalid task_name in response: {result.get('task_name')}")
                result["task_name"] = _keyword_fallback(command)
            return result

        except Exception as e:
            last_err = e
            err_str  = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = 60
                m = re.search(r"retry[^\d]*(\d+)", err_str, re.I)
                if m: wait = int(m.group(1)) + 5
                if attempt < 2:
                    print(f"  [GEMINI] Rate-limited. Waiting {wait}s "
                          f"(attempt {attempt+2}/3)...")
                    time.sleep(wait)
                else:
                    print("  [GEMINI] Quota exhausted — keyword fallback.")
            else:
                print(f"  [GEMINI] Error: {e}"); break
    print(f"  [GEMINI] Falling back to keyword-based task: {task}")
    task = _keyword_fallback(command)
   
    return {"task_name": task,
            "scene_observation": f"(Gemini unavailable: {type(last_err).__name__})",
            "reasoning": "keyword fallback"}

def _keyword_fallback(cmd):
    cmd = cmd.lower()
    if any(w in cmd for w in ("hover","inspect","look")):             return "hover_all"
    if any(w in cmd for w in ("bowl","match")) and "stack" not in cmd: return "place_in_bowls"
    if any(w in cmd for w in ("line","row","arrange")):               return "line_up_by_color"
    if any(w in cmd for w in ("separate","each col","own")):          return "stack_by_color"
    return "stack_all"


# ── Core pick-and-place ───────────────────────────────────────────────────────
def vision_pick_and_place(cube_color, cube_idx, place_xyz, viewer):
    """
    All motion targets come from vision (camera unprojection).
    ctrl.move_to solves IK via Jacobian pseudoinverse.
    data.xpos used only to identify which physics body to weld.
    """
    cube_pos = vs.cube(cube_color, cube_idx)
    if cube_pos is None:
        print(f"  [SKIP] {cube_color}_cube_{cube_idx+1} not detected"); return

    # Weld body resolution: nearest physics body to vision XY (sim only)
    cube_body_id = min(
        CUBE_BODY_IDS[cube_color],
        key=lambda bid: np.linalg.norm(ctrl.data.xpos[bid][:2] - cube_pos[:2])
    )
    rname  = mujoco.mj_id2name(ctrl.model, mujoco.mjtObj.mjOBJ_BODY, cube_body_id)
    gt_pos = ctrl.data.xpos[cube_body_id]
    xy_err = np.linalg.norm(gt_pos[:2] - cube_pos[:2]) * 100

    print(f"\n  +-- PICK & PLACE ----------------------------------------+")
    print(f"  |  cube      : {cube_color}_cube_{cube_idx+1}")
    print(f"  |  vision xyz: {np.round(cube_pos,3)}  <- IK target (camera)")
    print(f"  |  place xyz : {np.round(place_xyz,3)}")
    print(f"  |  weld body : {rname}  vis_err={xy_err:.1f}cm  (weld only)")
    print(f"  +--------------------------------------------------------+")

    hover_pos      = cube_pos.copy();   hover_pos[2]      += ctrl.FINGER_OFFSET + ctrl.HOVER_HEIGHT
    grasp_pos      = cube_pos.copy();   grasp_pos[2]      += ctrl.FINGER_OFFSET + ctrl.GRASP_HEIGHT
    approach_place = place_xyz.copy(); approach_place[2]  += ctrl.FINGER_OFFSET + ctrl.HOVER_HEIGHT
    drop_place     = place_xyz.copy(); drop_place[2]      += ctrl.FINGER_OFFSET + 0.005

    success = False
    for attempt in range(3):
        if attempt > 0:
            print(f"    [RETRY] attempt {attempt+1}/3")
            ctrl.detach_all(); ctrl.open_gripper(viewer, 200); ctrl.go_home(viewer)

        ctrl.data.ctrl[:7] = ctrl.data.qpos[:7].copy()
        ctrl.open_gripper(viewer, 100)
        ctrl.detach_all()
        ctrl.move_to(hover_pos, viewer, max_steps=30000)
        ctrl.pause(viewer, 200)
        ctrl.descend_to(grasp_pos, viewer)
        ctrl.close_and_weld(cube_body_id, viewer)
        if ctrl.grasp_held(cube_body_id, viewer):
            success = True; break
        ctrl.detach_all(); ctrl.open_gripper(viewer, 100)

    if not success:
        print(f"    [ABORT] {rname} after 3 attempts")
        ctrl.detach_all(); ctrl.reset_to_home()
        mujoco.mj_forward(ctrl.model, ctrl.data); viewer.sync(); return

    transit = ctrl.data.xpos[ctrl.hand_id].copy(); transit[2] += 0.30
    ctrl.move_to(transit, viewer)
    ctrl.move_to(approach_place, viewer)
    ctrl.move_to(drop_place, viewer)
    ctrl.detach_all(); ctrl.open_gripper(viewer, 200); ctrl.pause(viewer, 400)
    retract = ctrl.data.xpos[ctrl.hand_id].copy(); retract[2] += 0.20
    ctrl.move_to(retract, viewer)
    ctrl.go_home(viewer)


# ── Task implementations — all geometry from vision ───────────────────────────

def task_hover_all(viewer):
    """Hover over every detected cube. Positions 100% from vision."""
    for color in COLORS:
        for idx in range(2):
            pos = vs.cube(color, idx)
            if pos is None:
                print(f"  [SKIP] {color}_cube_{idx+1} not detected"); continue
            hover = pos.copy(); hover[2] += ctrl.FINGER_OFFSET + ctrl.HOVER_HEIGHT
            print(f"  [HOVER] {color}_cube_{idx+1} vision={np.round(pos,3)}")
            ctrl.move_to(hover, viewer); ctrl.pause(viewer, 500)
    ctrl.go_home(viewer)


def task_stack_all(viewer):
    """
    Stack all cubes into one tower.
    Drop XY = centroid of all detected cube XY positions (recomputed each pick).
    Drop Z  = increments by ctrl.CUBE_HEIGHT per layer.
    """
    layer = 0
    for color in COLORS:
        for idx in range(2):
            if layer % REDETECT_EVERY == 0: vs.update(viewer)
            drop_xy = vs.scene_centroid()   # vision-derived, no hardcoded XY
            target  = np.array([drop_xy[0], drop_xy[1],
                                 ctrl.STACK_BASE_Z + layer * ctrl.CUBE_HEIGHT])
            print(f"  [STACK_ALL] layer={layer}  "
                  f"drop_xy(vision centroid)={np.round(drop_xy,3)}")
            vision_pick_and_place(color, idx, target, viewer)
            layer += 1


def task_stack_by_color(viewer):
    """
    Three separate towers, one per color.
    Tower XY = centroid of that color's detected cubes, shifted outward in Y.
    Spread derived from actual Y-spread seen in the scene.
    """
    vs.update(viewer)
    scene_y = vs.scene_centroid()[1]
    tower_positions = {}
    for color in COLORS:
        c = vs.color_centroid(color)
        if c is None: c = vs.scene_centroid()
        # Push tower outward in Y from scene centre so it's in clear space
        dy = np.sign(c[1] - scene_y) * 0.12 if abs(c[1] - scene_y) > 0.02 else 0.0
        tower_positions[color] = np.array([c[0], c[1] + dy])
        print(f"  [STACK_BY_COLOR] {color} tower "
              f"(vision-derived) = {np.round(tower_positions[color],3)}")

    op = 0
    for color in COLORS:
        for layer, idx in enumerate(range(2)):
            if op % REDETECT_EVERY == 0: vs.update(viewer)
            xy     = tower_positions[color]
            target = np.array([xy[0], xy[1],
                                ctrl.STACK_BASE_Z + layer * ctrl.CUBE_HEIGHT])
            vision_pick_and_place(color, idx, target, viewer)
            op += 1


def task_place_in_bowls(viewer):
    """
    Place each cube into the matching-color bowl.
    Bowl position comes entirely from vision detection.
    """
    op = 0
    for color in COLORS:
        for layer, idx in enumerate(range(2)):
            if op % REDETECT_EVERY == 0: vs.update(viewer)
            bowl_pos = vs.bowl(color)
            if bowl_pos is None:
                print(f"  [SKIP] {color} bowl not detected"); op += 1; continue
            print(f"  [BOWL] {color} bowl vision pos: {np.round(bowl_pos,3)}")
            target = np.array([bowl_pos[0], bowl_pos[1],
                                BOWL_Z + layer * ctrl.CUBE_HEIGHT])
            vision_pick_and_place(color, idx, target, viewer)
            op += 1


def task_line_up_by_color(viewer):
    """
    Line cubes in rows by color.
    X: just behind the forward-most detected cube (free space from vision).
    Y: divided from the scene's actual Y-spread (no hardcoded values).
    Cube spacing: derived from Y-spread / number of positions.
    """
    vs.update(viewer)
    x_min, x_max, y_min, y_max = vs.scene_bounds()
    x_drop      = max(0.35, x_min - 0.08)          # free space behind cubes
    y_span      = max(y_max - y_min, 0.20)          # at least 20cm
    n_rows      = len(COLORS)
    row_height  = y_span / n_rows
    cube_gap    = row_height / 2.0

    print(f"  [LINE_UP] x_drop={x_drop:.3f}  "
          f"y_range=[{y_min:.3f},{y_max:.3f}]  "
          f"row_height={row_height:.3f}  cube_gap={cube_gap:.3f}")

    op = 0
    for ri, color in enumerate(COLORS):
        y0 = y_min + ri * row_height
        for i, idx in enumerate(range(2)):
            if op % REDETECT_EVERY == 0: vs.update(viewer)
            target = np.array([x_drop, y0 + i * cube_gap, ctrl.STACK_BASE_Z])
            print(f"  [LINE_UP] {color}_cube_{idx+1} -> {np.round(target,3)}")
            vision_pick_and_place(color, idx, target, viewer)
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


# ── Command runner ────────────────────────────────────────────────────────────
def run_command(command, viewer):
    print(f"\n{'='*62}")
    print(f"  COMMAND : {command}")
    print(f"{'='*62}")

    print("\n  [1/3] Camera capture + vision detection...")
    rgb, det = capture_and_detect()
    vs._det  = det
    print_detections(det)

    print("  [2/3] Gemini vision-language planning...")
    t0      = time.time()
    result  = query_gemini(command, rgb)
    task    = result["task_name"]
    latency = time.time() - t0

    print(f"\n  +-- Gemini ---------------------------------------------+")
    print(f"  |  model   : {GEMINI_MODEL}")
    print(f"  |  task    : {task}")
    print(f"  |  scene   : {result.get('scene_observation','--')[:55]}")
    print(f"  |  latency : {latency:.2f}s")
    print(f"  +-------------------------------------------------------+\n")

    print(f"  [3/3] Executing '{task}' via IK (targets from camera)...")
    ctrl.reset_to_home(); mujoco.mj_forward(ctrl.model, ctrl.data); viewer.sync()
    time.sleep(0.4)
    VISION_TASK_MAP[task](viewer)
    ctrl.reset_to_home(); mujoco.mj_forward(ctrl.model, ctrl.data); viewer.sync()
    print(f"\n  Done -- '{task}' complete.")

    result["_meta"] = {
        "camera":           CAMERA_NAME,
        "img_size":         f"{IMG_W}x{IMG_H}",
        "gemini_model":     GEMINI_MODEL,
        "gemini_latency_s": round(latency, 2),
        "redetect_every":   REDETECT_EVERY,
    }
    return result


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_config()

    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"
    with mujoco.viewer.launch_passive(ctrl.model, ctrl.data) as viewer:
        viewer.cam.lookat[:] = [0.45, 0.0, 0.3]
        viewer.cam.distance  = 1.8
        viewer.cam.elevation = -20
        ctrl.reset_to_home(); mujoco.mj_forward(ctrl.model, ctrl.data); viewer.sync()
        time.sleep(1.0)
        log = []

        if mode == "interactive":
            print("\nInteractive VLA -- type a command, 'demo', 'reset', or 'quit'.\n")
            while viewer.is_running():
                try:    cmd = input("  > ").strip()
                except: break
                if not cmd: continue
                if cmd.lower() in ("quit", "q"): break
                if cmd.lower() == "demo":
                    for c in DEMO_COMMANDS: log.append(run_command(c, viewer))
                elif cmd.lower() == "reset":
                    ctrl.reset_to_home()
                    mujoco.mj_forward(ctrl.model, ctrl.data); viewer.sync()
                    print("  [RESET] Home position restored.")
                else:
                    log.append(run_command(cmd, viewer))
        else:
            for i, cmd in enumerate(DEMO_COMMANDS, 1):
                print(f"\n[{i}/{len(DEMO_COMMANDS)}]")
                log.append(run_command(cmd, viewer))
                if i < len(DEMO_COMMANDS): time.sleep(1.0)

        if log:
            p = f"vla_vision_log_{int(time.time())}.json"
            json.dump(log, open(p, "w"), indent=2)
            print(f"\n  Log saved -> {p}")

        print("\nDone. Ctrl-C to exit.")
        while viewer.is_running():
            mujoco.mj_step(ctrl.model, ctrl.data); viewer.sync()
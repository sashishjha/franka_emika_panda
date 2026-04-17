# ik_controller_pick_and_put.py
# mjpython ik_controller_pick_and_put.py

import mujoco
import mujoco.viewer
import numpy as np
import os

# ── Config ───────────────────────────────────────────────────────────────────
SCENE_XML    = os.path.join(os.path.dirname(__file__), "scene.xml")

HOVER_HEIGHT  = 0.15
GRASP_HEIGHT  = 0.05
LAMBDA        = 0.05
GAIN          = 1.2
MAX_DCTRL     = 0.008
POS_TOL       = 0.005
IK_EVERY      = 10
FINGER_OFFSET = 0.10  # offset from hand center to finger tips (for better grasping)
HEADLESS      = True

# ── Load model ────────────────────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path(SCENE_XML)
data  = mujoco.MjData(model)

print(f"Timestep: {model.opt.timestep}s")

hand_id       = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
red_cube_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")
blue_cube_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "blue_cube")
green_cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "green_cube")
weld_id       = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld")

print(f"hand={hand_id}  red={red_cube_id}  blue={blue_cube_id}  green={green_cube_id}  weld={weld_id}")

# ── Home pose ─────────────────────────────────────────────────────────────────
HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

def reset_to_home(data):
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = HOME
    data.ctrl[:7] = HOME
    data.ctrl[7]  = 255
    model.eq_active0[weld_id] = 0  # make sure weld is off at start
    mujoco.mj_forward(model, data)

reset_to_home(data)

print("Warming up physics…")
for _ in range(2000):
    mujoco.mj_step(model, data)

# ── IK step ───────────────────────────────────────────────────────────────────
def ik_step(model, data, hand_id, target_pos):
    nv       = model.nv
    hand_pos = data.xpos[hand_id].copy()
    error    = target_pos - hand_pos
    err_norm = np.linalg.norm(error)

    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)
    J = jacp[:, :7]

    A  = J @ J.T + (LAMBDA ** 2) * np.eye(3)
    dq = GAIN * J.T @ np.linalg.solve(A, error)

    if err_norm > 0.1:
        dq *= 0.3

    dq = np.clip(dq, -MAX_DCTRL, MAX_DCTRL)
    data.ctrl[:7] = data.ctrl[:7] + dq

    return err_norm

# ── Motion primitives ─────────────────────────────────────────────────────────
def get_pos(body_id, z_offset=0.0):
    return data.xpos[body_id].copy() + np.array([0.0, 0.0, z_offset + FINGER_OFFSET])

def move_to(target_pos, viewer=None, max_steps=20000):
    err = 999.0
    for i in range(max_steps):
        if viewer and not viewer.is_running():
            break
        if i % IK_EVERY == 0:
            err = ik_step(model, data, hand_id, target_pos)
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()
        if err < POS_TOL:
            break
    print(f"    arrived  err={err:.4f}m")

def open_gripper(viewer=None, steps=400):
    data.ctrl[7] = 255
    model.eq_active0[weld_id] = 0   # ← release weld
    for _ in range(steps):
        if viewer and not viewer.is_running():
            break
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()

def close_gripper(viewer=None, steps=400):
    data.ctrl[7] = 0
    model.eq_active0[weld_id] = 1   # ← activate weld
    for _ in range(steps):
        if viewer and not viewer.is_running():
            break
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()

def attach_cube(cube_id):
    model.eq_obj2id[weld_id] = cube_id
    model.eq_active0[weld_id] = 1   # ← fixed

def detach_cube():
    model.eq_active0[weld_id] = 0   # ← fixed

def go_home(viewer=None, steps=800):
    data.ctrl[:7] = HOME
    for _ in range(steps):
        if viewer and not viewer.is_running():
            break
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()

# ── Pick and place ────────────────────────────────────────────────────────────
def pick_and_place(cube_id, place_id, viewer=None):
    cube_name  = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, cube_id)
    place_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, place_id)
    print(f"\n── Pick {cube_name} → place on {place_name} ──")

    # 1. Make sure weld off, open gripper, hover above cube
    print("  1. Hover above cube")
    detach_cube()
    open_gripper(viewer, steps=200)
    move_to(get_pos(cube_id, HOVER_HEIGHT), viewer)

    # 2. Descend to grasp height
    print("  2. Descend to grasp")
    move_to(get_pos(cube_id, GRASP_HEIGHT), viewer)

    # 3. Attach weld + close gripper
    print("  3. Grasp")
    attach_cube(cube_id)
    data.ctrl[7] = 0
    if viewer:
        for _ in range(300):
            mujoco.mj_step(model, data)
            viewer.sync()

    # 4. Lift straight up
    print("  4. Lift")
    lift_target = data.xpos[hand_id].copy()
    lift_target[2] += 0.25
    move_to(lift_target, viewer)

    # 5. Move above place target
    print("  5. Move above place target")
    place_pos = get_pos(place_id)
    place_z   = place_pos[2] + 0.08
    move_to(np.array([place_pos[0], place_pos[1], place_z + HOVER_HEIGHT]), viewer)

    # 6. Descend to place
    print("  6. Descend to place")
    move_to(np.array([place_pos[0], place_pos[1], place_z + 0.01]), viewer)

    # 7. Release
    print("  7. Release")
    detach_cube()
    open_gripper(viewer, steps=500)

    # 8. Lift away
    print("  8. Lift away")
    lift_target = data.xpos[hand_id].copy()
    lift_target[2] += 0.2
    move_to(lift_target, viewer)

    # 9. Go home
    print("  9. Return home")
    go_home(viewer)

# ── Run sequence ──────────────────────────────────────────────────────────────
def run_sequence(viewer=None):
    print("\n═══ SEQUENCE START ═══")
    print("Task: Pick red → place on green, then pick blue → place on green")

    pick_and_place(red_cube_id, green_cube_id, viewer)

    if viewer:
        for _ in range(500):
            mujoco.mj_step(model, data)
            viewer.sync()

    pick_and_place(blue_cube_id, green_cube_id, viewer)

    print("\n═══ SEQUENCE COMPLETE ═══")

# ── Main ──────────────────────────────────────────────────────────────────────
if HEADLESS:
    run_sequence(viewer=None)
else:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.5, 0.0, 0.3]
        viewer.cam.distance   = 1.6
        viewer.cam.elevation  = -20
        run_sequence(viewer)
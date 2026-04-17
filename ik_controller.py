# ik_controller_vla.py
# Run with: mjpython ik_controller_vla.py

import mujoco
import mujoco.viewer
import numpy as np
import os

SCENE_XML     = os.path.join(os.path.dirname(__file__), "scene.xml")

HOVER_HEIGHT  = 0.25
GRASP_HEIGHT  = 0.010
CUBE_HEIGHT   = 0.065
STACK_BASE_Z  = 0.035
LAMBDA        = 0.01
GAIN_POS      = 0.85
GAIN_ORI      = 3.0
DAMP_VEL      = 0.3
MAX_DCTRL     = 0.05
POS_TOL       = 0.008
ORI_TOL       = 0.10
IK_EVERY      = 5
FINGER_OFFSET = 0.10
DESCENT_STEP  = 0.003
GRIPPER_OPEN        = 255
GRIPPER_CLOSE_FIRM  = 10
GRIPPER_STEP        = 3

DESIRED_ZAXIS = np.array([0.0, 0.0, -1.0])

JOINT_LOW  = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_HIGH = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

# ══════════════════════════════════════════════════════════════
model = mujoco.MjModel.from_xml_path(SCENE_XML)
data  = mujoco.MjData(model)
print(f"Timestep: {model.opt.timestep}s")

def bid(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
def eid(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)

hand_id = bid("hand")

CUBES = {
    "red":   [bid("red_cube_1"),   bid("red_cube_2")],
    "blue":  [bid("blue_cube_1"),  bid("blue_cube_2")],
    "green": [bid("green_cube_1"), bid("green_cube_2")],
}
ALL_CUBES = CUBES["red"] + CUBES["blue"] + CUBES["green"]

BOWLS = {
    "red":   bid("red_bowl"),
    "blue":  bid("blue_bowl"),
    "green": bid("green_bowl"),
}

WELDS = {
    bid("red_cube_1"):   eid("weld_red1"),
    bid("red_cube_2"):   eid("weld_red2"),
    bid("blue_cube_1"):  eid("weld_blue1"),
    bid("blue_cube_2"):  eid("weld_blue2"),
    bid("green_cube_1"): eid("weld_green1"),
    bid("green_cube_2"): eid("weld_green2"),
}

HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

def reset_to_home():
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = HOME
    data.ctrl[:7] = HOME
    data.ctrl[7]  = GRIPPER_OPEN
    for wid in WELDS.values():
        model.eq_active0[wid] = 0
        data.eq_active[wid]   = 0
    mujoco.mj_forward(model, data)

reset_to_home()
print("Warming up physics...")
for _ in range(2000):
    mujoco.mj_step(model, data)
print(f"Gripper ctrl=255 → qpos[7]={data.qpos[7]:.4f}  (expect ~0.04 = open)")

# ══════════════════════════════════════════════════════════════
#  IK
# ══════════════════════════════════════════════════════════════
def ik_step(target_pos, step_limit=None):
    hand_pos  = data.xpos[hand_id].copy()
    pos_error = target_pos - hand_pos
    R          = data.xmat[hand_id].reshape(3, 3)
    ori_error  = np.cross(R[:, 2], DESIRED_ZAXIS)
    error = np.concatenate([GAIN_POS * pos_error, GAIN_ORI * ori_error])

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)
    J = np.vstack([jacp[:, :7], jacr[:, :7]])
    A      = J @ J.T + (LAMBDA**2) * np.eye(6)
    J_pinv = J.T @ np.linalg.solve(A, np.eye(6))
    dq_task = J_pinv @ error

    nullspace_proj = np.eye(7) - (J_pinv @ J)
    q          = data.qpos[:7]
    q_mid      = (JOINT_LOW + JOINT_HIGH) / 2.0
    q_range    = JOINT_HIGH - JOINT_LOW
    limit_grad = -2.0 * (q - q_mid) / (q_range ** 2)
    pos_norm   = np.linalg.norm(pos_error)
    dq_null    = nullspace_proj @ (0.08 * (HOME - q) + 0.04 * limit_grad)
    vel_damp   = -DAMP_VEL * data.qvel[:7]
    dq = dq_task + dq_null + vel_damp
    if pos_norm > 0.1:
        dq *= 0.3

    max_step = MAX_DCTRL if step_limit is None else float(step_limit)
    dq_norm = np.linalg.norm(dq)
    if dq_norm > max_step:
        dq = dq * (max_step / dq_norm)
    return dq, pos_norm, np.linalg.norm(ori_error)

# ══════════════════════════════════════════════════════════════
#  PRIMITIVES
# ══════════════════════════════════════════════════════════════
def body_pos(body_id):
    return data.xpos[body_id].copy()

def above(body_id, extra_z=0.0):
    p = body_pos(body_id).copy()
    p[2] += FINGER_OFFSET + extra_z
    return p

def move_to(target_pos, viewer, max_steps=30000, dctrl_override=None):
    step_limit = MAX_DCTRL if dctrl_override is None else float(dctrl_override)
    pos_err = ori_err = 999.0
    for i in range(max_steps):
        if not viewer.is_running():
            break
        if i % IK_EVERY == 0:
            dq, pos_err, ori_err = ik_step(target_pos, step_limit=step_limit)
            data.ctrl[:7] = np.clip(data.ctrl[:7] + dq, JOINT_LOW, JOINT_HIGH)
        mujoco.mj_step(model, data)
        if i % 5 == 0:
            viewer.sync()
        if pos_err < POS_TOL and ori_err < ORI_TOL:
            break
        if i == max_steps - 1:
            print(f"    [WARNING] move_to timeout  pos={pos_err:.4f}  ori={ori_err:.4f}")
    pause(viewer, 50)
    return pos_err, ori_err

def pause(viewer, steps=400):
    for _ in range(steps):
        if not viewer.is_running():
            break
        mujoco.mj_step(model, data)
        viewer.sync()

def open_gripper(viewer, steps=300):
    data.ctrl[7] = GRIPPER_OPEN
    pause(viewer, steps)

def go_home(viewer, steps=1500):
    data.ctrl[:7] = data.qpos[:7].copy()
    for i in range(steps):
        if not viewer.is_running():
            break
        alpha = (i + 1) / steps
        data.ctrl[:7] = (1.0 - alpha) * data.qpos[:7] + alpha * HOME
        mujoco.mj_step(model, data)
        if i % 5 == 0:
            viewer.sync()
    # Hard snap to kill windup
    data.ctrl[:7] = HOME.copy()
    pause(viewer, 200)

def detach_all():
    for wid in WELDS.values():
        model.eq_active0[wid] = 0
        data.eq_active[wid]   = 0

def descend_to(target_pos, viewer):
    hand_now = data.xpos[hand_id].copy()
    dz = target_pos[2] - hand_now[2]
    n_waypoints = max(1, int(abs(dz) / DESCENT_STEP))
    for i in range(1, n_waypoints + 1):
        alpha = i / n_waypoints
        wp = hand_now + alpha * (target_pos - hand_now)
        move_to(wp, viewer, max_steps=3000, dctrl_override=0.02)
    return move_to(target_pos, viewer, max_steps=8000, dctrl_override=0.015)

# ══════════════════════════════════════════════════════════════
#  WELD — correct relpose computation using mju_mat2Quat
# ══════════════════════════════════════════════════════════════
def attach_after_close(cube_id):
    """
    Compute hand→cube relative transform at the moment of contact,
    then activate the weld with that exact relpose.

    MuJoCo weld eq_data[0:7] = [px, py, pz, qw, qx, qy, qz]
    where (p, q) is the pose of body2 (cube) in body1 (hand) frame.
    """
    detach_all()
    mujoco.mj_forward(model, data)

    hand_pos = data.xpos[hand_id].copy()
    hand_mat = data.xmat[hand_id].reshape(3, 3)
    cube_pos = data.xpos[cube_id].copy()
    cube_mat = data.xmat[cube_id].reshape(3, 3)

    # Position of cube in hand frame
    rel_pos = hand_mat.T @ (cube_pos - hand_pos)

    # Orientation of cube in hand frame
    rel_mat = hand_mat.T @ cube_mat
    rel_quat = np.zeros(4)
    mujoco.mju_mat2Quat(rel_quat, rel_mat.flatten())

    wid = WELDS[cube_id]
    model.eq_data[wid, 0:3] = rel_pos
    model.eq_data[wid, 3:7] = rel_quat

    model.eq_active0[wid] = 1
    data.eq_active[wid]   = 1
    mujoco.mj_forward(model, data)
    print(f"    [WELD] rel_pos={np.round(rel_pos,4)}  |r|={np.linalg.norm(rel_pos):.4f}")

# ══════════════════════════════════════════════════════════════
#  CONTACT SENSING
# ══════════════════════════════════════════════════════════════
def _geoms_for_body(body_id):
    start = model.body_geomadr[body_id]
    count = model.body_geomnum[body_id]
    return set(range(start, start + count))

_FINGER_GEOMS = set()
for _i in range(model.nbody):
    _bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, _i) or ""
    if any(kw in _bname.lower() for kw in ["finger", "pad", "panda_hand", "gripper"]):
        _FINGER_GEOMS |= _geoms_for_body(_i)

if not _FINGER_GEOMS:
    print("[WARNING] No finger geoms found — stall detection only")
else:
    print(f"Finger geom IDs: {_FINGER_GEOMS}")

def fingers_touching_cube(cube_id, require_both=False):
    cube_geoms = _geoms_for_body(cube_id)
    hits = 0
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if (g1 in _FINGER_GEOMS and g2 in cube_geoms) or \
           (g2 in _FINGER_GEOMS and g1 in cube_geoms):
            hits += 1
    return hits >= (2 if require_both else 1)

# ══════════════════════════════════════════════════════════════
#  CLOSE GRIPPER → WELD
# ══════════════════════════════════════════════════════════════
def close_and_weld(cube_id, viewer, max_iter=120):
    current   = float(data.ctrl[7])
    min_ctrl  = float(GRIPPER_CLOSE_FIRM)
    prev_fpos = data.qpos[7]
    welded    = False

    for step in range(max_iter):
        if not viewer.is_running():
            break
        current      = max(current - GRIPPER_STEP, min_ctrl)
        data.ctrl[7] = current

        contact_found = False
        for inner in range(8):
            mujoco.mj_step(model, data)
            if _FINGER_GEOMS and fingers_touching_cube(cube_id, require_both=False):
                print(f"    [GRIPPER] Geom contact ctrl={current:.0f} inner={inner}")
                attach_after_close(cube_id)
                pause(viewer, 150)
                welded = True
                contact_found = True
                break
        if contact_found:
            break
        viewer.sync()

        curr_fpos = data.qpos[7]
        if step > 4:
            finger_moved = abs(curr_fpos - prev_fpos)
            if finger_moved < 0.0003 and current < (GRIPPER_OPEN - 30):
                print(f"    [GRIPPER] Stall ctrl={current:.0f}  delta={finger_moved:.5f}")
                attach_after_close(cube_id)
                pause(viewer, 150)
                welded = True
                break
        prev_fpos = curr_fpos

    if not welded:
        print(f"    [GRIPPER] No contact after {max_iter} iters — force-close weld")
        data.ctrl[7] = min_ctrl
        pause(viewer, 200)
        attach_after_close(cube_id)
        welded = True

    return welded

# ══════════════════════════════════════════════════════════════
#  GRASP VERIFICATION — delta-based, not absolute distance
# ══════════════════════════════════════════════════════════════
def grasp_held(cube_id, viewer, lift_dist=0.04):
    """
    Lift slowly. A successful grasp means the cube's Z rises with the hand.
    We measure follow_ratio = cube_dz / hand_dz; threshold > 0.5.
    This is immune to weld offset magnitude.
    """
    pause(viewer, 400)
    hand_before = data.xpos[hand_id].copy()
    cube_before = data.xpos[cube_id].copy()

    lift_target = hand_before.copy()
    lift_target[2] += lift_dist
    move_to(lift_target, viewer, max_steps=10000, dctrl_override=0.008)
    pause(viewer, 100)

    hand_after = data.xpos[hand_id].copy()
    cube_after = data.xpos[cube_id].copy()

    hand_dz = hand_after[2] - hand_before[2]
    cube_dz = cube_after[2] - cube_before[2]
    follow_ratio = (cube_dz / hand_dz) if hand_dz > 0.005 else 0.0

    cname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, cube_id)
    print(f"    [GRASP CHECK] {cname}  hand_dz={hand_dz:.4f}  "
          f"cube_dz={cube_dz:.4f}  follow={follow_ratio:.2f}")
    return follow_ratio > 0.5

# ══════════════════════════════════════════════════════════════
#  PICK & PLACE
# ══════════════════════════════════════════════════════════════
def pick_and_place_to(cube_id, place_xyz, viewer):
    place_xyz = np.array(place_xyz, dtype=float)
    cname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, cube_id)
    print(f"\n  pick_and_place: {cname} → {np.round(place_xyz, 3)}")

    for attempt in range(3):
        if attempt > 0:
            print(f"    [RETRY] attempt {attempt + 1}/3")
            detach_all()
            open_gripper(viewer, 200)
            go_home(viewer, 1500)

        data.ctrl[:7] = data.qpos[:7].copy()

        open_gripper(viewer, 100)
        detach_all()

        hover_target = above(cube_id, HOVER_HEIGHT)
        move_to(hover_target, viewer, max_steps=30000)
        pause(viewer, 300)

        grasp_target = above(cube_id, GRASP_HEIGHT)
        descend_to(grasp_target, viewer)

        hand_z    = data.xpos[hand_id][2]
        cube_z    = data.xpos[cube_id][2]
        tip_z     = hand_z - FINGER_OFFSET
        gap       = tip_z - (cube_z + 0.03)
        R         = data.xmat[hand_id].reshape(3, 3)
        ori_align = np.dot(R[:, 2], DESIRED_ZAXIS)
        print(f"    [PRE-GRASP] tip_z={tip_z:.4f}  cube_top={cube_z+0.03:.4f}"
              f"  gap={gap:+.4f}  ori_align={ori_align:.3f}")

        if gap > 0.015:
            extra = grasp_target.copy()
            extra[2] -= (gap - 0.008)
            extra[2]  = max(extra[2], cube_z + 0.028)
            move_to(extra, viewer, max_steps=8000, dctrl_override=0.01)
            tip_z = data.xpos[hand_id][2] - FINGER_OFFSET
            gap   = tip_z - (cube_z + 0.03)
            print(f"    [NUDGE]     tip_z={tip_z:.4f}  gap={gap:+.4f}")

        close_and_weld(cube_id, viewer)

        if grasp_held(cube_id, viewer):
            break
        else:
            detach_all()
            open_gripper(viewer, 100)
            if attempt == 2:
                print(f"    [ABORT] Could not grasp {cname} — skipping")
                go_home(viewer, 800)
                return

    hand_now = data.xpos[hand_id].copy()
    transit  = hand_now.copy()
    transit[2] = hand_now[2] + 0.22
    move_to(transit, viewer)

    approach    = place_xyz.copy()
    approach[2] = place_xyz[2] + FINGER_OFFSET + HOVER_HEIGHT
    move_to(approach, viewer)

    drop    = place_xyz.copy()
    drop[2] = place_xyz[2] + FINGER_OFFSET + 0.01
    move_to(drop, viewer)

    detach_all()
    open_gripper(viewer, 200)
    pause(viewer, 400)

    retract    = data.xpos[hand_id].copy()
    retract[2] += 0.20
    move_to(retract, viewer)
    go_home(viewer, 800)

# ══════════════════════════════════════════════════════════════
#  TASKS
# ══════════════════════════════════════════════════════════════
def hover_all(viewer):
    print("\n-- TASK 1: Hover and inspect all cubes --")
    for cube_id in ALL_CUBES:
        cname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, cube_id)
        print(f"  Hovering: {cname}")
        move_to(above(cube_id, HOVER_HEIGHT), viewer)
        pause(viewer, 500)
    go_home(viewer)
    print("  OK")

STACK_ALL_XY = np.array([0.50, 0.0])

def stack_all(viewer):
    print("\n-- TASK 2: Stack ALL cubes into one tower --")
    for layer, cube_id in enumerate(ALL_CUBES):
        z = STACK_BASE_Z + layer * CUBE_HEIGHT
        pick_and_place_to(cube_id, [STACK_ALL_XY[0], STACK_ALL_XY[1], z], viewer)
    print("  OK")

COLOR_TOWER_XY = {
    "red":   np.array([0.60, 0.30]),
    "blue":  np.array([0.45, 0.30]),
    "green": np.array([0.30, 0.30]),
}

def stack_by_color(viewer):
    print("\n-- TASK 3: Stack each color into its own tower --")
    for color, cube_ids in CUBES.items():
        print(f"  Building {color} tower...")
        for layer, cube_id in enumerate(cube_ids):
            z  = STACK_BASE_Z + layer * CUBE_HEIGHT
            xy = COLOR_TOWER_XY[color]
            pick_and_place_to(cube_id, [xy[0], xy[1], z], viewer)
    print("  OK")

def place_in_bowls(viewer):
    print("\n-- TASK 4: Place each cube in its matching bowl --")
    for color, cube_ids in CUBES.items():
        bowl_id  = BOWLS[color]
        bowl_xy  = body_pos(bowl_id)[:2]
        for layer, cube_id in enumerate(cube_ids):
            z = STACK_BASE_Z + layer * CUBE_HEIGHT
            pick_and_place_to(cube_id, [bowl_xy[0], bowl_xy[1], z], viewer)
    print("  OK")

LINE_X   = 0.55
LINE_GAP = 0.10
COLOR_LINE_Y = {"red": 0.30, "blue": 0.15, "green": 0.00}

def line_up_by_color(viewer):
    print("\n-- TASK 5: Line up each color's cubes --")
    for color, cube_ids in CUBES.items():
        y0 = COLOR_LINE_Y[color]
        for i, cube_id in enumerate(cube_ids):
            pick_and_place_to(cube_id, [LINE_X, y0 + i * LINE_GAP, STACK_BASE_Z], viewer)
    print("  OK")

TASK_MAP = {
    "hover_all":        hover_all,
    "stack_all":        stack_all,
    "stack_by_color":   stack_by_color,
    "place_in_bowls":   place_in_bowls,
    "line_up_by_color": line_up_by_color,
}

# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.45, 0.0, 0.3]
        viewer.cam.distance   = 1.8
        viewer.cam.elevation  = -20

        print("\n=== STANDALONE TEST: all 5 tasks ===\n")

        hover_all(viewer)
        reset_to_home(); pause(viewer, 500)

        stack_all(viewer)
        reset_to_home(); pause(viewer, 500)

        stack_by_color(viewer)
        reset_to_home(); pause(viewer, 500)

        place_in_bowls(viewer)
        reset_to_home(); pause(viewer, 500)

        line_up_by_color(viewer)

        print("\nAll 5 tasks complete!")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
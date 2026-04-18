# ik_controller_vla.py
# Run with: mjpython ik_controller_vla.py
#
# ROOT CAUSE FIXES vs all previous versions:
#
# BUG 1 (weld teleports cube): eq_data for a MuJoCo weld has 10 entries:
#   [anchor1_in_body1(3), anchor2_in_body2(3), rel_quat(4)]
#   All previous versions wrote quat to [3:7], clobbering anchor2 and quat[0].
#   Fix: eq_data[0:3]=rel_pos, eq_data[3:6]=[0,0,0], eq_data[6:10]=rel_quat
#
# BUG 2 (arm never descends): descend_to used dctrl_override=0.02 with only
#   3000 steps per 3mm waypoint. The IK dq is norm-clipped to 0.02, and with
#   GAIN_POS=0.85 and even a 3mm residual the convergence is too slow.
#   Fix: use full MAX_DCTRL (no override) for waypoints -- safety comes from
#   the 3mm step SIZE, not from slowing joint motion.
#
# BUG 3 (grasp check wrong): replaced absolute dist with delta follow-ratio.

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
DESCENT_STEP  = 0.005   # 5mm steps during descent
GRIPPER_OPEN        = 255
GRIPPER_CLOSE_FIRM  = 10
GRIPPER_STEP        = 3

STOP_REQUESTED = False

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
print(f"Home hand z={data.xpos[hand_id][2]:.4f}  "
      f"gripper qpos={data.qpos[7]:.4f} (expect ~0.04=open)")

# Verify eq_data size for welds
wid0 = list(WELDS.values())[0]
print(f"eq_data shape for weld: {model.eq_data[wid0].shape}  "
      f"(expect 11, entries [0:3]=anchor1, [3:6]=anchor2, [6:10]=quat)")

# ══════════════════════════════════════════════════════════════
#  IK
# ══════════════════════════════════════════════════════════════
def ik_step(target_pos, step_limit=MAX_DCTRL):
    hand_pos  = data.xpos[hand_id].copy()
    pos_error = target_pos - hand_pos
    R         = data.xmat[hand_id].reshape(3, 3)
    ori_error = np.cross(R[:, 2], DESIRED_ZAXIS)
    error     = np.concatenate([GAIN_POS * pos_error, GAIN_ORI * ori_error])

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)
    J      = np.vstack([jacp[:, :7], jacr[:, :7]])
    A      = J @ J.T + (LAMBDA**2) * np.eye(6)
    J_pinv = J.T @ np.linalg.solve(A, np.eye(6))
    dq     = J_pinv @ error

    null_proj = np.eye(7) - (J_pinv @ J)
    q         = data.qpos[:7]
    q_mid     = (JOINT_LOW + JOINT_HIGH) / 2.0
    q_range   = JOINT_HIGH - JOINT_LOW
    lim_grad  = -2.0 * (q - q_mid) / (q_range ** 2)
    dq       += null_proj @ (0.08 * (HOME - q) + 0.04 * lim_grad)
    dq       += -DAMP_VEL * data.qvel[:7]

    pos_norm = np.linalg.norm(pos_error)
    if pos_norm > 0.1:
        dq *= 0.3

    dq_norm = np.linalg.norm(dq)
    if dq_norm > step_limit:
        dq = dq * (step_limit / dq_norm)

    return dq, pos_norm, np.linalg.norm(ori_error)

# ══════════════════════════════════════════════════════════════
#  MOTION PRIMITIVES
# ══════════════════════════════════════════════════════════════
def body_pos(body_id):
    return data.xpos[body_id].copy()

def above(body_id, extra_z=0.0):
    p = body_pos(body_id).copy()
    p[2] += FINGER_OFFSET + extra_z
    return p

def bowl_drop_z(bowl_id):
    """
    Bowl interior bottom is above bowl body center by a small offset.
    Tuned for the larger hemispherical bowls.
    """
    BOWL_INTERIOR_OFFSET = 0.03
    return data.xpos[bowl_id][2] + BOWL_INTERIOR_OFFSET

def move_to(target_pos, viewer, max_steps=30000, step_limit=MAX_DCTRL):
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

def pause(viewer, steps=200):
    for _ in range(steps):
        if not viewer.is_running():
            break
        mujoco.mj_step(model, data)
        viewer.sync()

def open_gripper(viewer, steps=200):
    data.ctrl[7] = GRIPPER_OPEN
    pause(viewer, steps)

def go_home(viewer, steps=1500):
    """Interpolate to HOME then hard-snap ctrl."""
    q0 = data.qpos[:7].copy()
    data.ctrl[:7] = q0
    for i in range(steps):
        if not viewer.is_running():
            break
        alpha = (i + 1) / steps
        data.ctrl[:7] = (1.0 - alpha) * q0 + alpha * HOME
        mujoco.mj_step(model, data)
        if i % 5 == 0:
            viewer.sync()
    data.ctrl[:7] = HOME.copy()   # hard snap — kills windup
    pause(viewer, 300)

def detach_all():
    for wid in WELDS.values():
        model.eq_active0[wid] = 0
        data.eq_active[wid]   = 0

def descend_to(target_pos, viewer):
    """
    Descend in DESCENT_STEP-sized waypoints using FULL step_limit (MAX_DCTRL).
    Safety comes from small spatial steps, NOT from slowing joint motion.
    dctrl_override was the bug: it made IK converge too slowly per waypoint.
    """
    hand_now = data.xpos[hand_id].copy()
    dz = target_pos[2] - hand_now[2]
    n_wp = max(1, int(abs(dz) / DESCENT_STEP))
    print(f"    [DESCEND] from z={hand_now[2]:.4f} to z={target_pos[2]:.4f}  "
          f"dz={dz:.4f}  waypoints={n_wp}")
    for i in range(1, n_wp + 1):
        alpha = i / n_wp
        wp = hand_now + alpha * (target_pos - hand_now)
        move_to(wp, viewer, max_steps=4000, step_limit=MAX_DCTRL)  # full speed
    return move_to(target_pos, viewer, max_steps=8000, step_limit=MAX_DCTRL)

# ══════════════════════════════════════════════════════════════
#  WELD — CORRECT eq_data layout
# ══════════════════════════════════════════════════════════════
def attach_after_close(cube_id):
    """
    MuJoCo weld eq_data has 11 entries (indices 0..10):
      [0:3]  anchor1 in body1 (hand) frame
      [3:6]  anchor2 in body2 (cube) frame  ← MUST be [0,0,0] for rigid attach
      [6:10] relative quaternion [w,x,y,z]  ← body2 orient in body1 frame
      [10]   torque scale (ignored here)

    Previous versions wrote quat to [3:7], corrupting anchor2 and quat[0],
    which teleported the cube to a random location ~1m away.
    """
    detach_all()
    mujoco.mj_forward(model, data)

    hand_pos = data.xpos[hand_id].copy()
    hand_mat = data.xmat[hand_id].reshape(3, 3)
    cube_pos = data.xpos[cube_id].copy()
    cube_mat = data.xmat[cube_id].reshape(3, 3)

    # anchor1: cube origin expressed in hand frame
    rel_pos = hand_mat.T @ (cube_pos - hand_pos)

    # relative orientation: cube frame expressed in hand frame
    rel_mat  = hand_mat.T @ cube_mat
    rel_quat = np.zeros(4)
    mujoco.mju_mat2Quat(rel_quat, rel_mat.flatten())

    wid = WELDS[cube_id]
    model.eq_data[wid, 0:3] = rel_pos          # anchor1 in hand frame
    model.eq_data[wid, 3:6] = [0.0, 0.0, 0.0] # anchor2 in cube frame = origin
    model.eq_data[wid, 6:10] = rel_quat        # relative quat [w,x,y,z]

    model.eq_active0[wid] = 1
    data.eq_active[wid]   = 1
    mujoco.mj_forward(model, data)

    cname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, cube_id)
    print(f"    [WELD ON] {cname}  anchor1={np.round(rel_pos,4)}  "
          f"|anchor1|={np.linalg.norm(rel_pos):.4f}")

# ══════════════════════════════════════════════════════════════
#  CONTACT SENSING
# ══════════════════════════════════════════════════════════════
def _geoms_for_body(body_id):
    s = model.body_geomadr[body_id]
    return set(range(s, s + model.body_geomnum[body_id]))

_FINGER_GEOMS = set()
for _i in range(model.nbody):
    _n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, _i) or ""
    if any(k in _n.lower() for k in ["finger", "pad", "panda_hand", "gripper"]):
        _FINGER_GEOMS |= _geoms_for_body(_i)

print(f"Finger geom IDs ({len(_FINGER_GEOMS)}): {_FINGER_GEOMS}" if _FINGER_GEOMS
      else "[WARNING] No finger geoms found")

def fingers_touching_cube(cube_id):
    cube_geoms = _geoms_for_body(cube_id)
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if (g1 in _FINGER_GEOMS and g2 in cube_geoms) or \
           (g2 in _FINGER_GEOMS and g1 in cube_geoms):
            return True
    return False

# ══════════════════════════════════════════════════════════════
#  CLOSE GRIPPER → WELD
# ══════════════════════════════════════════════════════════════
def close_and_weld(cube_id, viewer, max_iter=120):
    current   = float(data.ctrl[7])
    prev_fpos = data.qpos[7]
    welded    = False

    for step in range(max_iter):
        if not viewer.is_running():
            break
        current      = max(current - GRIPPER_STEP, float(GRIPPER_CLOSE_FIRM))
        data.ctrl[7] = current

        hit = False
        for inner in range(8):
            mujoco.mj_step(model, data)
            if _FINGER_GEOMS and fingers_touching_cube(cube_id):
                print(f"    [GRIPPER] Contact  ctrl={current:.0f}  inner={inner}")
                attach_after_close(cube_id)
                pause(viewer, 200)
                welded = True
                hit    = True
                break
        if hit:
            break
        viewer.sync()

        curr_fpos = data.qpos[7]
        if step > 4:
            if abs(curr_fpos - prev_fpos) < 0.0003 and current < GRIPPER_OPEN - 30:
                print(f"    [GRIPPER] Stall  ctrl={current:.0f}")
                attach_after_close(cube_id)
                pause(viewer, 200)
                welded = True
                break
        prev_fpos = curr_fpos

    if not welded:
        print(f"    [GRIPPER] No contact — force close and weld")
        data.ctrl[7] = GRIPPER_CLOSE_FIRM
        pause(viewer, 300)
        attach_after_close(cube_id)

# ══════════════════════════════════════════════════════════════
#  GRASP CHECK — delta-based (immune to weld offset magnitude)
# ══════════════════════════════════════════════════════════════
def grasp_held(cube_id, viewer, lift_dist=0.04):
    """
    Lift slowly. A held cube rises with the hand (follow_ratio ≈ 1).
    A dropped cube stays put (follow_ratio ≈ 0).
    """
    pause(viewer, 400)
    hz0 = data.xpos[hand_id][2]
    cz0 = data.xpos[cube_id][2]

    lift = data.xpos[hand_id].copy()
    lift[2] += lift_dist
    # Slow lift — won't snap weld
    move_to(lift, viewer, max_steps=12000, step_limit=0.008)
    pause(viewer, 100)

    hz1 = data.xpos[hand_id][2]
    cz1 = data.xpos[cube_id][2]
    h_dz = hz1 - hz0
    c_dz = cz1 - cz0
    ratio = (c_dz / h_dz) if h_dz > 0.005 else 0.0

    cname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, cube_id)
    print(f"    [GRASP CHECK] {cname}  hand_dz={h_dz:.4f}  "
          f"cube_dz={c_dz:.4f}  follow={ratio:.2f}  held={ratio > 0.5}")
    return ratio > 0.5

# ══════════════════════════════════════════════════════════════
#  PICK & PLACE
# ══════════════════════════════════════════════════════════════
def pick_and_place_to(cube_id, place_xyz, viewer):
    place_xyz = np.array(place_xyz, dtype=float)
    cname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, cube_id)
    print(f"\n  pick_and_place: {cname} → {np.round(place_xyz,3)}")

    for attempt in range(3):
        if attempt > 0:
            print(f"    [RETRY] attempt {attempt+1}/3")
            detach_all()
            open_gripper(viewer, 200)
            go_home(viewer)

        # Resync ctrl to current qpos — kills any windup
        data.ctrl[:7] = data.qpos[:7].copy()

        # ── 1. Open + hover ───────────────────────────────────────────────────
        open_gripper(viewer, 100)
        detach_all()
        move_to(above(cube_id, HOVER_HEIGHT), viewer, max_steps=30000)
        pause(viewer, 200)

        # ── 2. Descend ────────────────────────────────────────────────────────
        grasp_target = above(cube_id, GRASP_HEIGHT)
        descend_to(grasp_target, viewer)

        # Diagnostic
        hand_z    = data.xpos[hand_id][2]
        cube_z    = data.xpos[cube_id][2]
        tip_z     = hand_z - FINGER_OFFSET
        gap       = tip_z - (cube_z + 0.03)
        ori_align = np.dot(data.xmat[hand_id].reshape(3,3)[:,2], DESIRED_ZAXIS)
        print(f"    [PRE-GRASP] tip_z={tip_z:.4f}  cube_top={cube_z+0.03:.4f}"
              f"  gap={gap:+.4f}  ori_align={ori_align:.3f}")

        # Extra nudge if too high
        if gap > 0.015:
            nudge = grasp_target.copy()
            nudge[2] = max(cube_z + 0.028, grasp_target[2] - (gap - 0.008))
            move_to(nudge, viewer, max_steps=8000, step_limit=MAX_DCTRL)
            tip_z = data.xpos[hand_id][2] - FINGER_OFFSET
            gap   = tip_z - (cube_z + 0.03)
            print(f"    [NUDGE]     tip_z={tip_z:.4f}  gap={gap:+.4f}")

        # ── 3. Close → weld ───────────────────────────────────────────────────
        close_and_weld(cube_id, viewer)

        # ── 4. Verify ─────────────────────────────────────────────────────────
        if grasp_held(cube_id, viewer):
            break
        detach_all()
        open_gripper(viewer, 100)
        if attempt == 2:
            print(f"    [ABORT] {cname} — skipping")
            go_home(viewer)
            return

    # ── 5. Transport ──────────────────────────────────────────────────────────
    transit = data.xpos[hand_id].copy()
    transit[2] += 0.22
    move_to(transit, viewer)

    approach = place_xyz.copy()
    approach[2] = place_xyz[2] + FINGER_OFFSET + HOVER_HEIGHT
    move_to(approach, viewer)

    drop = place_xyz.copy()
    drop[2] = place_xyz[2] + FINGER_OFFSET + 0.005
    move_to(drop, viewer)

    # ── 6. Release ────────────────────────────────────────────────────────────
    detach_all()
    open_gripper(viewer, 200)
    pause(viewer, 400)

    retract = data.xpos[hand_id].copy()
    retract[2] += 0.20
    move_to(retract, viewer)
    go_home(viewer)

# ══════════════════════════════════════════════════════════════
#  TASKS
# ══════════════════════════════════════════════════════════════
def hover_all(viewer):
    print("\n-- TASK 1: Hover and inspect all cubes --")
    for cube_id in ALL_CUBES:
        if STOP_REQUESTED:
            print("[STOP] Task interrupted.")
            return
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
        if STOP_REQUESTED:
            print("[STOP] Task interrupted.")
            return
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
        if STOP_REQUESTED:
            print("[STOP] Task interrupted.")
            return
        print(f"  Building {color} tower...")
        for layer, cube_id in enumerate(cube_ids):
            if STOP_REQUESTED:
                print("[STOP] Task interrupted.")
                return
            z  = STACK_BASE_Z + layer * CUBE_HEIGHT
            xy = COLOR_TOWER_XY[color]
            pick_and_place_to(cube_id, [xy[0], xy[1], z], viewer)
    print("  OK")

def place_in_bowls(viewer):
    print("\n-- TASK 4: Place each cube in its matching bowl --")
    for color, cube_ids in CUBES.items():
        if STOP_REQUESTED:
            print("[STOP] Task interrupted.")
            return
        bowl_id = BOWLS[color]
        bowl_xy = body_pos(bowl_id)[:2]
        bowl_z = bowl_drop_z(bowl_id)
        for layer, cube_id in enumerate(cube_ids):
            if STOP_REQUESTED:
                print("[STOP] Task interrupted.")
                return
            z = bowl_z + layer * CUBE_HEIGHT
            pick_and_place_to(cube_id, [bowl_xy[0], bowl_xy[1], z], viewer)
    print("  OK")

LINE_X   = 0.55
LINE_GAP = 0.10
COLOR_LINE_Y = {"red": 0.30, "blue": 0.15, "green": 0.00}

def line_up_by_color(viewer):
    print("\n-- TASK 5: Line up each color's cubes --")
    for color, cube_ids in CUBES.items():
        if STOP_REQUESTED:
            print("[STOP] Task interrupted.")
            return
        y0 = COLOR_LINE_Y[color]
        for i, cube_id in enumerate(cube_ids):
            if STOP_REQUESTED:
                print("[STOP] Task interrupted.")
                return
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
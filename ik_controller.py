# ik_controller_vla.py
# Run with: mjpython ik_controller_vla.py

import mujoco
import mujoco.viewer
import numpy as np
import os

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
SCENE_XML     = os.path.join(os.path.dirname(__file__), "scene.xml")

HOVER_HEIGHT  = 0.15
GRASP_HEIGHT  = 0.005
CUBE_HEIGHT   = 0.065
STACK_BASE_Z  = 0.035
LAMBDA        = 0.05
GAIN_POS      = 1.2
GAIN_ORI      = 1.0
MAX_DCTRL     = 0.003
POS_TOL       = 0.006
ORI_TOL       = 0.05
IK_EVERY      = 10
FINGER_OFFSET = 0.10

DESIRED_ZAXIS = np.array([0.0, 0.0, -1.0])

# ══════════════════════════════════════════════════════════════
#  LOAD MODEL
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

print("Body IDs:", {k: v for k, v in [("hand", hand_id)] +
      [(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i), i)
       for i in ALL_CUBES]})

# ══════════════════════════════════════════════════════════════
#  HOME POSE
# ══════════════════════════════════════════════════════════════
HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

def reset_to_home():
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = HOME
    data.ctrl[:7] = HOME
    data.ctrl[7]  = 255
    for wid in WELDS.values():
        model.eq_active0[wid] = 0
    mujoco.mj_forward(model, data)

reset_to_home()
print("Warming up physics...")
for _ in range(2000):
    mujoco.mj_step(model, data)

# ══════════════════════════════════════════════════════════════
#  6-DOF IK CORE  (position + orientation)
#
#  Standard 3-DOF IK controls only WHERE the hand goes.
#  This 6-DOF version also controls HOW the hand is oriented,
#  forcing a top-down (perpendicular) approach to every cube.
#
#  Orientation error:
#    hand_zaxis = R[:, 2]  (third column of hand rotation matrix)
#    ori_error  = cross(hand_zaxis, desired_zaxis)
#    This is zero when hand_zaxis is parallel to desired_zaxis.
#    desired_zaxis = [0,0,-1] means gripper points straight down.
#
#  Full Jacobian: stack jacp (3xnv) and jacr (3xnv) -> (6xnv)
#  DLS: A = J @ J.T + lambda^2 * I_6   (6x6 system instead of 3x3)
# ══════════════════════════════════════════════════════════════
def ik_step(target_pos):
    hand_pos  = data.xpos[hand_id].copy()
    pos_error = target_pos - hand_pos

    R          = data.xmat[hand_id].reshape(3, 3)
    hand_zaxis = R[:, 2]
    ori_error  = np.cross(hand_zaxis, DESIRED_ZAXIS)

    error = np.concatenate([
        GAIN_POS * pos_error,
        GAIN_ORI * ori_error
    ])

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)
    J = np.vstack([jacp[:, :7], jacr[:, :7]])   # (6, 7)

    # --- Upgraded DLS with Nullspace Projection ---
    # 1. Damped pseudo-inverse
    A = J @ J.T + (LAMBDA**2) * np.eye(6)
    J_pinv = J.T @ np.linalg.solve(A, np.eye(6))
    
    # 2. Primary task: move hand to target
    dq_task = J_pinv @ error
    
    # 3. Secondary task: pull elbow/joints back to HOME pose
    I = np.eye(7)
    nullspace_proj = I - (J_pinv @ J)
    q_error = HOME - data.qpos[:7]
    dq_null = nullspace_proj @ (0.05 * q_error)  # 0.05 is the pull strength
    
    # Combine both tasks
    dq = dq_task + dq_null
    # ----------------------------------------------

    if np.linalg.norm(pos_error) > 0.1:
        dq *= 0.3

    dq = np.clip(dq, -MAX_DCTRL, MAX_DCTRL)
    
    # THE WINDUP FIX: Anchor command to physical state
    data.ctrl[:7] = data.qpos[:7] + dq

    return np.linalg.norm(pos_error), np.linalg.norm(ori_error)

# ══════════════════════════════════════════════════════════════
#  MOTION PRIMITIVES
# ══════════════════════════════════════════════════════════════
def body_pos(body_id):
    return data.xpos[body_id].copy()

def above(body_id, extra_z=0.0):
    p = body_pos(body_id).copy()
    p[2] += FINGER_OFFSET + extra_z
    return p

def move_to(target_pos, viewer, max_steps=20000):
    pos_err = 999.0
    ori_err = 999.0
    for i in range(max_steps):
        if not viewer.is_running():
            break
        if i % IK_EVERY == 0:
            pos_err, ori_err = ik_step(target_pos)
        mujoco.mj_step(model, data)
        if i % 5 == 0:
            viewer.sync()
        if pos_err < POS_TOL and ori_err < ORI_TOL:
            break
        
        # Add this warning
        if i == max_steps - 1:
            print(f"    [WARNING] move_to timed out! pos_err={pos_err:.4f}, ori_err={ori_err:.4f}")
            
    return pos_err, ori_err

def pause(viewer, steps=400):
    for _ in range(steps):
        if not viewer.is_running():
            break
        mujoco.mj_step(model, data)
        viewer.sync()

def open_gripper(viewer, steps=300):
    data.ctrl[7] = 255
    pause(viewer, steps)

def close_gripper(viewer, steps=300):
    data.ctrl[7] = 0
    pause(viewer, steps)

def go_home(viewer, steps=800):
    data.ctrl[:7] = HOME
    pause(viewer, steps)

def attach(cube_id):
    for wid in WELDS.values():
        model.eq_active0[wid] = 0
        data.eq_active[wid]   = 0
    model.eq_active0[WELDS[cube_id]] = 1
    data.eq_active[WELDS[cube_id]]   = 1

def detach_all():
    for wid in WELDS.values():
        model.eq_active0[wid] = 0
        data.eq_active[wid]   = 0

def _log_grasp_gap(cube_id):
    hand_z      = data.xpos[hand_id][2]
    cube_z      = data.xpos[cube_id][2]
    fingertip_z = hand_z - FINGER_OFFSET
    cube_top_z  = cube_z + 0.03
    gap         = fingertip_z - cube_top_z
    R           = data.xmat[hand_id].reshape(3, 3)
    hand_zaxis  = R[:, 2]
    ori_align   = np.dot(hand_zaxis, DESIRED_ZAXIS)
    cname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, cube_id)
    print(f"    [GRASP DIAG] fingertip_z={fingertip_z:.4f}  cube_top_z={cube_top_z:.4f}"
          f"  gap={gap:+.4f}m  ori_align={ori_align:.3f} (ideal=-1.0)  ({cname})")

# ══════════════════════════════════════════════════════════════
#  PICK & PLACE
# ══════════════════════════════════════════════════════════════
def pick_and_place_to(cube_id, place_xyz, viewer):
    global MAX_DCTRL
    place_xyz = np.array(place_xyz, dtype=float)
    cname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, cube_id)
    print(f"  -> pick {cname} -> {np.round(place_xyz, 3)}")

    detach_all()
    open_gripper(viewer, 100)

    # 1. Hover
    move_to(above(cube_id, HOVER_HEIGHT), viewer)
    
    # 2. Slow staged descent — gripper stays wide open the whole time
    #    so fingers never touch cube walls during approach
    MAX_DCTRL = 0.002
    data.ctrl[7] = 255                                               # enforce open
    move_to(above(cube_id, 0.08), viewer, max_steps=30000)
    data.ctrl[7] = 255                                               # enforce open
    move_to(above(cube_id, 0.03), viewer, max_steps=30000)
    data.ctrl[7] = 255                                               # enforce open
    move_to(above(cube_id, GRASP_HEIGHT), viewer, max_steps=30000)
    
    # 3. Confirm position before grasping
    _log_grasp_gap(cube_id)
    gap = data.xpos[hand_id][2] - FINGER_OFFSET - (data.xpos[cube_id][2] + 0.03)
    if abs(gap) > 0.025:
        # Not close enough — try one more descent
        print(f"    [GRASP] gap too large ({gap:.4f}m), nudging down...")
        move_to(above(cube_id, GRASP_HEIGHT - 0.01), viewer, max_steps=30000)
    
    # 4. Now close gripper slowly around the cube, THEN activate weld
    # 4. Activate weld FIRST to lock physics, THEN close gripper visually
    attach(cube_id)
    pause(viewer, 50)       # let weld settle
    
    data.ctrl[7] = 180      # close partway
    pause(viewer, 100)
    data.ctrl[7] = 0        # fully closed
    pause(viewer, 200)
    
    MAX_DCTRL = 0.003
    
   

    # 4. Lift
    hand_now       = data.xpos[hand_id].copy()
    lift_target    = hand_now.copy()
    lift_target[2] = hand_now[2] + 0.28
    move_to(lift_target, viewer)

    # 5. Move above place XY
    approach    = place_xyz.copy()
    approach[2] = place_xyz[2] + FINGER_OFFSET + HOVER_HEIGHT
    move_to(approach, viewer)

    # 6. Descend to drop height
    drop    = place_xyz.copy()
    drop[2] = place_xyz[2] + FINGER_OFFSET + 0.01
    move_to(drop, viewer)

    # 7. Release
    detach_all()
    open_gripper(viewer, 200)
    pause(viewer, 300)   # let cube settle

    # 8. Retract
    retract    = data.xpos[hand_id].copy()
    retract[2] += 0.20
    move_to(retract, viewer)

    go_home(viewer, 600)

# ══════════════════════════════════════════════════════════════
#  TASK 1 — hover_all
# ══════════════════════════════════════════════════════════════
def hover_all(viewer):
    print("\n-- TASK 1: Hover and inspect all cubes --")
    for cube_id in ALL_CUBES:
        cname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, cube_id)
        print(f"  Hovering over {cname}...")
        move_to(above(cube_id, HOVER_HEIGHT), viewer)
        pause(viewer, steps=500)
    go_home(viewer)
    print("  OK Inspection complete")

# ══════════════════════════════════════════════════════════════
#  TASK 2 — stack_all
# ══════════════════════════════════════════════════════════════
STACK_ALL_XY = np.array([0.50, 0.0])

def stack_all(viewer):
    print("\n-- TASK 2: Stack ALL cubes into one tower --")
    for layer, cube_id in enumerate(ALL_CUBES):
        z         = STACK_BASE_Z + layer * CUBE_HEIGHT
        place_xyz = np.array([STACK_ALL_XY[0], STACK_ALL_XY[1], z])
        pick_and_place_to(cube_id, place_xyz, viewer)
    print("  OK Single tower complete")

# ══════════════════════════════════════════════════════════════
#  TASK 3 — stack_by_color
# ══════════════════════════════════════════════════════════════
COLOR_TOWER_XY = {
    "red":   np.array([0.60,  0.30]),
    "blue":  np.array([0.45,  0.30]),
    "green": np.array([0.30,  0.30]),
}

def stack_by_color(viewer):
    print("\n-- TASK 3: Stack each color into its own tower --")
    for color, cube_ids in CUBES.items():
        print(f"  Building {color} tower...")
        for layer, cube_id in enumerate(cube_ids):
            z         = STACK_BASE_Z + layer * CUBE_HEIGHT
            xy        = COLOR_TOWER_XY[color]
            place_xyz = np.array([xy[0], xy[1], z])
            pick_and_place_to(cube_id, place_xyz, viewer)
    print("  OK Three color towers complete")

# ══════════════════════════════════════════════════════════════
#  TASK 4 — place_in_bowls
# ══════════════════════════════════════════════════════════════
def place_in_bowls(viewer):
    print("\n-- TASK 4: Place each cube in its matching bowl --")
    for color, cube_ids in CUBES.items():
        bowl_id  = BOWLS[color]
        bowl_xy  = body_pos(bowl_id)[:2]
        for layer, cube_id in enumerate(cube_ids):
            z         = STACK_BASE_Z + layer * CUBE_HEIGHT
            place_xyz = np.array([bowl_xy[0], bowl_xy[1], z])
            pick_and_place_to(cube_id, place_xyz, viewer)
    print("  OK All cubes in bowls")

# ══════════════════════════════════════════════════════════════
#  TASK 5 — line_up_by_color
# ══════════════════════════════════════════════════════════════
LINE_X   = 0.55
LINE_GAP = 0.10

COLOR_LINE_Y = {
    "red":   0.30,
    "blue":  0.15,
    "green": 0.00,
}

def line_up_by_color(viewer):
    print("\n-- TASK 5: Line up each color's cubes in a row --")
    for color, cube_ids in CUBES.items():
        y0 = COLOR_LINE_Y[color]
        print(f"  Lining up {color} cubes at y~{y0:.2f}...")
        for i, cube_id in enumerate(cube_ids):
            place_xyz = np.array([LINE_X, y0 + i * LINE_GAP, STACK_BASE_Z])
            pick_and_place_to(cube_id, place_xyz, viewer)
    print("  OK All rows arranged")

# ══════════════════════════════════════════════════════════════
#  EXPORT
# ══════════════════════════════════════════════════════════════
TASK_MAP = {
    "hover_all":        hover_all,
    "stack_all":        stack_all,
    "stack_by_color":   stack_by_color,
    "place_in_bowls":   place_in_bowls,
    "line_up_by_color": line_up_by_color,
}

# ══════════════════════════════════════════════════════════════
#  STANDALONE TEST
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.45, 0.0, 0.3]
        viewer.cam.distance   = 1.8
        viewer.cam.elevation  = -20

        print("\n====================================")
        print("  VLA CONTROLLER -- standalone test  ")
        print("====================================\n")

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
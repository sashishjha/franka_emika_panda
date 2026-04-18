# calibrate_hsv.py  — fixed world_to_pixel projection
# Run with: mjpython calibrate_hsv.py
# Requires: camera pos="0.45 0.0 1.0" euler="0 0 0" fovy="90" in scene.xml

import numpy as np
import cv2
import mujoco
import ik_controller_vla as ctrl

CAMERA = "cam_top"
IMG_H = IMG_W = 224
SAMPLE_R = 8   # px radius to sample around projected point

cam_id = mujoco.mj_name2id(ctrl.model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA)
assert cam_id >= 0, f"Camera '{CAMERA}' not found"

for _ in range(500):
    mujoco.mj_step(ctrl.model, ctrl.data)

renderer = mujoco.Renderer(ctrl.model, height=IMG_H, width=IMG_W)
renderer.update_scene(ctrl.data, camera=CAMERA)
rgb = renderer.render().copy()
hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

cv2.imwrite("calibrate_raw.jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
print("Raw render saved → calibrate_raw.jpg")

# Print cam_xmat so we can verify it's identity for euler="0 0 0"
cam_mat = ctrl.data.cam_xmat[cam_id].reshape(3, 3)
print(f"\ncam_xmat (should be identity for euler='0 0 0'):\n{cam_mat.round(3)}\n")


def world_to_pixel(world_pos):
    """
    Correct MuJoCo pinhole projection.
    Camera looks along -Z_cam. cam_mat columns = cam axes in world.
    p_cam = cam_mat.T @ (world - cam_pos)
    u = f*p_cam[0]/depth + cx
    v = f*(-p_cam[1])/depth + cy   ← single Y flip only
    """
    cam_pos = ctrl.data.cam_xpos[cam_id].copy()
    cam_mat = ctrl.data.cam_xmat[cam_id].reshape(3, 3).copy()
    fovy    = ctrl.model.cam_fovy[cam_id] * np.pi / 180.0
    f       = (IMG_H / 2.0) / np.tan(fovy / 2.0)
    cx, cy  = IMG_W / 2.0, IMG_H / 2.0

    p_cam = cam_mat.T @ (np.array(world_pos) - cam_pos)
    if p_cam[2] >= 0:
        return None, "behind camera"

    depth = -p_cam[2]
    u = f * p_cam[0] / depth + cx
    v = f * (-p_cam[1]) / depth + cy   # one flip: cam Y up → image Y down

    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < IMG_W and 0 <= vi < IMG_H):
        return None, f"outside image px=({ui},{vi})"
    return (ui, vi), "ok"


def sample_hsv_patch(u, v, r=SAMPLE_R):
    u0, u1 = max(0, u-r), min(IMG_W, u+r)
    v0, v1 = max(0, v-r), min(IMG_H, v+r)
    patch   = hsv[v0:v1, u0:u1]
    if patch.size == 0:
        return None
    return {
        "H": int(np.median(patch[:,:,0])),
        "S": int(np.median(patch[:,:,1])),
        "V": int(np.median(patch[:,:,2])),
    }


OBJECTS = {
    "red_cube_1":   "red",
    "red_cube_2":   "red",
    "blue_cube_1":  "blue",
    "blue_cube_2":  "blue",
    "green_cube_1": "green",
    "green_cube_2": "green",
    "red_bowl":     "red",
    "blue_bowl":    "blue",
    "green_bowl":   "green",
}

color_H = {"red": [], "blue": [], "green": []}
color_S = {"red": [], "blue": [], "green": []}
debug   = rgb.copy()

print(f"{'Object':<16} {'World XYZ':>26}  {'Pixel':>10}  {'H':>4} {'S':>4} {'V':>4}")
print("─" * 72)

for name, color in OBJECTS.items():
    bid = mujoco.mj_name2id(ctrl.model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        print(f"{name:<16}  NOT IN MODEL"); continue

    world = ctrl.data.xpos[bid]
    px, status = world_to_pixel(world)

    if px is None:
        print(f"{name:<16}  [{world[0]:.3f},{world[1]:.3f},{world[2]:.3f}]"
              f"  {status}")
        continue

    u, v = px
    vals = sample_hsv_patch(u, v)
    if vals is None:
        print(f"{name:<16}  px=({u},{v})  SAMPLE FAILED"); continue

    color_H[color].append(vals["H"])
    color_S[color].append(vals["S"])
    print(f"{name:<16}  [{world[0]:.3f},{world[1]:.3f},{world[2]:.3f}]"
          f"  px=({u:3d},{v:3d})  H={vals['H']:3d} S={vals['S']:3d} V={vals['V']:3d}")

    c = {"red":(0,0,255),"blue":(255,100,0),"green":(0,200,0)}[color]
    cv2.circle(debug, (u, v), SAMPLE_R, c, 2)
    cv2.putText(debug, name[:5], (u+SAMPLE_R+1, v),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, c, 1)

cv2.imwrite("calibrate_annotated.jpg",
            cv2.cvtColor(debug, cv2.COLOR_RGB2BGR))
print("\nAnnotated → calibrate_annotated.jpg")
print("Circles should sit ON TOP of the colored objects.")
print("If they're off → check cam_xmat printout above.\n")

# ── Generate HSV_RANGES ────────────────────────────────────────────────────────
print("\n── Paste into vision_detect.py ─────────────────────────────────────\n")
print("HSV_RANGES = {")
for color in ("red", "blue", "green"):
    hs = color_H[color]
    ss = color_S[color]
    if not hs:
        print(f'    "{color}": [],   # NO SAMPLES')
        continue
    h_med = int(np.median(hs))
    s_min = max(40, int(min(ss)) - 20)
    margin = 15

    if color == "red":
        # Red wraps at 0/180
        print(f'    "red": [')
        print(f'        (np.array([  0, {s_min}, 40]), np.array([ {min(10, h_med+margin)}, 255, 255])),')
        print(f'        (np.array([{max(160, h_med-margin)}, {s_min}, 40]), np.array([180, 255, 255])),')
        print(f'    ],')
    else:
        lo = max(0,   h_med - margin)
        hi = min(180, h_med + margin)
        print(f'    "{color}": [(np.array([{lo:3d}, {s_min:3d}, 40]), np.array([{hi:3d}, 255, 255])),],')

print("}")
print(f"\nH values found:")
for c in ("red","blue","green"):
    print(f"  {c}: {color_H[c]}")
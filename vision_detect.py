# vision_detect.py  — fixed projection math
import numpy as np
import cv2
import mujoco

TABLE_Z = 0.0
CUBE_Z  = TABLE_Z + 0.025   # cube center height
BOWL_Z  = TABLE_Z + 0.008   # bowl center height

# HSV ranges calibrated from your scene (euler="0 0 0", fovy=90)
HSV_RANGES = {
    "red": [
        (np.array([  0, 100, 80]), np.array([ 12, 255, 255])),
        (np.array([168, 100, 80]), np.array([180, 255, 255])),
    ],
    "blue":  [(np.array([110, 100, 80]), np.array([130, 255, 255])),],
    "green": [(np.array([ 50, 100, 80]), np.array([ 70, 255, 255])),],
}

MIN_BLOB_AREA = 30   # px² at 224×224


# ── Camera intrinsics ──────────────────────────────────────────────────────────
def get_intrinsics(model, cam_id, img_h, img_w):
    fovy = model.cam_fovy[cam_id] * np.pi / 180.0
    f    = (img_h / 2.0) / np.tan(fovy / 2.0)
    return {"f": f, "cx": img_w / 2.0, "cy": img_h / 2.0}


# ── Pixel → world XYZ (correct MuJoCo convention) ─────────────────────────────
def pixel_to_world(u, v, world_z, K, data, cam_id):
    """
    MuJoCo camera convention:
      cam_xmat columns = camera axes in world frame
      camera looks along  -Z_cam
      X_cam = right,  Y_cam = up  (image Y is DOWN, so flip)

    Ray in camera frame:
      ray_cam = [ (u-cx)/f,  -(v-cy)/f,  -1 ]
              ^^^^^^^^^^^^   ^^^^^^^^^
              X right        Y flipped (image down → cam up)
    """
    f, cx, cy = K["f"], K["cx"], K["cy"]
    cam_pos = data.cam_xpos[cam_id].copy()
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()  # world←cam

    ray_cam   = np.array([(u - cx) / f,
                          -(v - cy) / f,   # flip Y
                          -1.0])
    ray_world = cam_mat @ ray_cam           # rotate to world frame
    ray_world /= np.linalg.norm(ray_world)

    if abs(ray_world[2]) < 1e-8:
        return cam_pos.copy()
    t = (world_z - cam_pos[2]) / ray_world[2]
    return cam_pos + t * ray_world


# ── World → pixel (for calibration/debug) ─────────────────────────────────────
def world_to_pixel(world_pos, model, data, cam_id, img_h, img_w):
    """Project a 3D world point to pixel (u, v). Returns None if behind camera."""
    cam_pos = data.cam_xpos[cam_id].copy()
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()

    # Transform to camera frame: R_cam_from_world = cam_mat.T
    p_cam = cam_mat.T @ (world_pos - cam_pos)

    # Camera looks along -Z_cam; point must have p_cam[2] < 0 to be visible
    if p_cam[2] >= 0:
        return None

    fovy  = model.cam_fovy[cam_id] * np.pi / 180.0
    f     = (img_h / 2.0) / np.tan(fovy / 2.0)
    depth = -p_cam[2]   # positive

    u = f * ( p_cam[0] / depth) + img_w / 2.0
    v = f * (-p_cam[1] / depth) + img_h / 2.0   # flip Y (cam up → image down)

    return int(round(u)), int(round(v))


# ── Color segmentation ─────────────────────────────────────────────────────────
def _color_mask(hsv, color):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in HSV_RANGES[color]:
        mask |= cv2.inRange(hsv, lo, hi)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def _blobs(mask):
    """Return list of {u, v, area} for all blobs above MIN_BLOB_AREA."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_BLOB_AREA:
            continue
        M = cv2.moments(c)
        if M["m00"] < 1e-6:
            continue
        out.append({"u": M["m10"]/M["m00"], "v": M["m01"]/M["m00"], "area": area})
    return sorted(out, key=lambda b: -b["area"])


# ── Main detector ──────────────────────────────────────────────────────────────
class VisionDetector:
    def __init__(self, model, data, cam_id, img_h=224, img_w=224):
        self.model  = model
        self.data   = data
        self.cam_id = cam_id
        self.img_h  = img_h
        self.img_w  = img_w
        self.K      = get_intrinsics(model, cam_id, img_h, img_w)
        print(f"[VisionDetector] f={self.K['f']:.1f}px  "
              f"cx={self.K['cx']:.1f}  cy={self.K['cy']:.1f}")

    def detect(self, rgb, debug=False):
        """
        Returns:
          cubes: {color: [pos_or_None, pos_or_None]}
          bowls: {color: pos_or_None}
        
        Cube vs bowl classification: in a top-down view cubes are
        SMALLER blobs (solid square ≈ 8–15px wide) and bowls are
        LARGER blobs (ring/filled circle ≈ 15–30px wide).
        We classify: largest blob per color = bowl, next two = cubes.
        """
        hsv    = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        result = {"cubes": {}, "bowls": {}, "detected_counts": {}}
        dbg    = rgb.copy() if debug else None

        for color in ("red", "blue", "green"):
            mask  = _color_mask(hsv, color)
            blobs = _blobs(mask)

            # Classify by area: bowl is largest, cubes are next two
            bowl_blob  = blobs[0]  if len(blobs) >= 1 else None
            cube_blobs = blobs[1:3] if len(blobs) >= 2 else []

            # If only 1 blob and it's small, it's a cube not a bowl
            if len(blobs) == 1 and blobs[0]["area"] < 300:
                cube_blobs = [blobs[0]]
                bowl_blob  = None

            # Unproject bowl
            if bowl_blob:
                bowl_pos = pixel_to_world(bowl_blob["u"], bowl_blob["v"],
                                          BOWL_Z, self.K, self.data, self.cam_id)
                result["bowls"][color] = bowl_pos
            else:
                result["bowls"][color] = None

            # Unproject cubes
            cube_positions = []
            for cb in cube_blobs:
                cp = pixel_to_world(cb["u"], cb["v"],
                                    CUBE_Z, self.K, self.data, self.cam_id)
                cube_positions.append(cp)
            while len(cube_positions) < 2:
                cube_positions.append(None)
            result["cubes"][color] = cube_positions
            result["detected_counts"][color] = (len(cube_blobs), 1 if bowl_blob else 0)

            if debug and dbg is not None:
                c = {"red":(0,0,255),"blue":(255,100,0),"green":(0,180,0)}[color]
                for cb in cube_blobs:
                    cv2.circle(dbg, (int(cb["u"]),int(cb["v"])), 6, c, 2)
                    cv2.putText(dbg, f"{color[0]}c",
                                (int(cb["u"])+7,int(cb["v"])-7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
                if bowl_blob:
                    cv2.circle(dbg, (int(bowl_blob["u"]),int(bowl_blob["v"])), 10, c, 3)
                    cv2.putText(dbg, f"{color[0]}B",
                                (int(bowl_blob["u"])+11,int(bowl_blob["v"])-11),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)

        if debug:
            result["_debug_img"] = dbg
        return result


def print_detections(result):
    print("\n  ── Vision Detections ──────────────────────────────────")
    for color in ("red","blue","green"):
        for i, cp in enumerate(result["cubes"].get(color,[None,None])):
            s = f"[{cp[0]:.3f}, {cp[1]:.3f}, {cp[2]:.3f}]" if cp is not None else "NOT DETECTED"
            print(f"  {color}_cube_{i+1}: {s}")
        bp = result["bowls"].get(color)
        s  = f"[{bp[0]:.3f}, {bp[1]:.3f}, {bp[2]:.3f}]" if bp is not None else "NOT DETECTED"
        print(f"  {color}_bowl:   {s}")
    print("  ────────────────────────────────────────────────────────\n")


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    import ik_controller_vla as ctrl

    CAMERA = "cam_top"
    IMG_H = IMG_W = 224

    cam_id = mujoco.mj_name2id(ctrl.model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA)
    assert cam_id >= 0, f"Camera '{CAMERA}' not found"

    for _ in range(500):
        mujoco.mj_step(ctrl.model, ctrl.data)

    renderer = mujoco.Renderer(ctrl.model, height=IMG_H, width=IMG_W)
    renderer.update_scene(ctrl.data, camera=CAMERA)
    rgb = renderer.render().copy()

    detector = VisionDetector(ctrl.model, ctrl.data, cam_id, IMG_H, IMG_W)
    result   = detector.detect(rgb, debug=True)
    print_detections(result)

    print("  ── Ground truth vs vision XY error ─────────────────────")
    from scipy.spatial.distance import cdist    
    for color in ("red", "blue", "green"):
        gt_positions = []
        for bname in [f"{color}_cube_1", f"{color}_cube_2"]:
            bid = mujoco.mj_name2id(ctrl.model, mujoco.mjtObj.mjOBJ_BODY, bname)
            gt_positions.append(ctrl.data.xpos[bid][:2].copy()) 
        vis_positions = [vp[:2] for vp in result["cubes"][color] if vp is not None] 
        if len(vis_positions) == 0:
            print(f"  {color}: NO DETECTIONS")
            continue    
        gt_arr  = np.array(gt_positions)   # shape (2,2)
        vis_arr = np.array(vis_positions)  # shape (N,2)    
        D = cdist(vis_arr, gt_arr)  # (N,2) distance matrix
        for vi in range(len(vis_arr)):
            gi  = np.argmin(D[vi])
            err = D[vi][gi] * 100
            print(f"  {color}_cube_{vi+1}  "
                  f"vis=[{vis_arr[vi][0]:.3f},{vis_arr[vi][1]:.3f}]  "
                  f"best_gt=[{gt_arr[gi][0]:.3f},{gt_arr[gi][1]:.3f}]  "
                  f"err={err:.1f}cm")
    print("  ────────────────────────────────────────────────────────") 
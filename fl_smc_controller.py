# fl_smc_controller.py
# Final Project: FL+PD (Baseline) vs FL+SMC (Proposed) vs VLA (Gemini+SMC)
# Demonstrates SMC robustness under external disturbances (Report Sec 2.3)
# Run: mjpython fl_smc_controller.py

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer

# ══════════════════════════════════════════════════════════════════════════════
#  TRAJECTORY GENERATORS
# ══════════════════════════════════════════════════════════════════════════════
def circle_path(t, center):
    r, w = 0.12, 1.0
    c, s = np.cos(w*t), np.sin(w*t)
    return (center + np.array([r*c, r*s, 0.0]),
            np.array([-r*w*s, r*w*c, 0.0]),
            np.array([-r*w**2*c, -r*w**2*s, 0.0]))

def infinity_path(t, center):
    A, B, w = 0.14, 0.08, 1.0
    return (center + np.array([A*np.sin(w*t), B*np.sin(2*w*t), 0.0]),
            np.array([A*w*np.cos(w*t), 2*B*w*np.cos(2*w*t), 0.0]),
            np.array([-A*w**2*np.sin(w*t), -4*B*w**2*np.sin(2*w*t), 0.0]))

def _superellipse(t, ax, ay, k, w):
    c, s = np.cos(w*t), np.sin(w*t)
    tk = np.tanh(k)
    tkc, tks = np.tanh(k*c), np.tanh(k*s)
    sc2, ss2 = 1.0 - tkc**2, 1.0 - tks**2
    x, y   = ax*tkc/tk, ay*tks/tk
    dx     = -ax*k*w*s*sc2/tk
    dy     =  ay*k*w*c*ss2/tk
    ddx    = -ax*k*w**2*sc2*(c + 2*k*s**2*tkc)/tk
    ddy    = -ay*k*w**2*ss2*(s + 2*k*c**2*tks)/tk
    return np.array([x,y,0.0]), np.array([dx,dy,0.0]), np.array([ddx,ddy,0.0])

def square_path(t, center):
    p, v, a = _superellipse(t, 0.10, 0.10, 4.0, 0.8)
    return center+p, v, a

def rectangle_path(t, center):
    p, v, a = _superellipse(t, 0.14, 0.08, 4.0, 0.8)
    return center+p, v, a

TRAJ_MAP   = {"circle": circle_path, "infinity": infinity_path,
              "square": square_path, "rectangle": rectangle_path}
ALL_SHAPES = list(TRAJ_MAP.keys())
SHAPE_PROMPTS = {"circle":"Trace a circular path", "infinity":"Draw a figure-eight",
                 "square":"Draw a square shape", "rectangle":"Trace a rectangular path"}

# ══════════════════════════════════════════════════════════════════════════════
#  VLA INTEGRATION  (Gemini with retry)
# ══════════════════════════════════════════════════════════════════════════════
def vla_choose_trajectory(prompt, max_retries=3):
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("  [VLA] No API key. Defaulting to circle.")
        return "circle"
    print(f"  [VLA Gemini] Command: '{prompt}'")
    from google import genai
    client = genai.Client(api_key=api_key)
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=(f"User command: '{prompt}'. You control a robot arm. "
                          f"Available shapes: circle, infinity, square, rectangle. "
                          f"Reply with exactly ONE word."))
            word = resp.text.strip().lower()
            for sh in ALL_SHAPES:
                if sh in word:
                    print(f"  [VLA Gemini] Decision: {sh}")
                    return sh
            print(f"  [VLA Gemini] Decision: circle (parsed '{word}')")
            return "circle"
        except Exception as e:
            w = 2**attempt
            print(f"  [VLA] Attempt {attempt+1} failed ({e}). Retrying in {w}s...")
            time.sleep(w)
    print("  [VLA] All retries failed. Defaulting to circle.")
    return "circle"

# ══════════════════════════════════════════════════════════════════════════════
#  CONTROLLERS
# ══════════════════════════════════════════════════════════════════════════════
HOME    = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
TAU_MAX = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)
EE_OFFSET = np.array([0.0, 0.0, 0.1034])  # Z-offset from hand to fingertips

def get_ee_position(data, hand_id):
    return data.xpos[hand_id] + data.xmat[hand_id].reshape(3,3) @ EE_OFFSET

def get_jacobian(model, data, hand_id, point):
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, None, point, hand_id)
    return jacp[:, :7]

def _fl_core(model, data, hand_id, a_x, J_prev, dt, point):
    J = get_jacobian(model, data, hand_id, point)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    h = data.qfrc_bias[:7].copy()
    Jdqd = ((J - J_prev)/dt) @ data.qvel[:7]
    Jp = J.T @ np.linalg.solve(J@J.T + 1e-4*np.eye(3), np.eye(3))
    tau = M[:7,:7] @ Jp @ (a_x - Jdqd) + h
    N = np.eye(7) - Jp @ J
    tau += N @ (5.0*(HOME - data.qpos[:7]) - 0.7*data.qvel[:7])
    return np.clip(tau, -TAU_MAX, TAU_MAX), J

def fl_pd(model, data, hand_id, Xd, Xd_d, Xd_dd, J_prev, dt):
    X = get_ee_position(data, hand_id)
    J = get_jacobian(model, data, hand_id, X)
    e = Xd - X; ed = Xd_d - J@data.qvel[:7]
    a_x = Xd_dd + 40.0*ed + 400.0*e
    tau, Jo = _fl_core(model, data, hand_id, a_x, J_prev, dt, X)
    return tau, Jo, np.linalg.norm(e)

def fl_smc(model, data, hand_id, Xd, Xd_d, Xd_dd, J_prev, dt):
    X = get_ee_position(data, hand_id)
    J = get_jacobian(model, data, hand_id, X)
    e = Xd - X; ed = Xd_d - J@data.qvel[:7]
    s = ed + 10.0*e
    a_x = Xd_dd + 10.0*ed + 15.0*np.clip(s/0.02, -1, 1) + 5.0*s
    tau, Jo = _fl_core(model, data, hand_id, a_x, J_prev, dt, X)
    return tau, Jo, np.linalg.norm(e)

CTRL_MAP = {"PD": fl_pd, "SMC": fl_smc}

# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
SCENE_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene_torque.xml")

def make_arm_transparent(model, alpha=1.0):
    # Left empty so arm remains solid
    pass

def reset_robot(model, data):
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = HOME.copy()
    mujoco.mj_forward(model, data)
    for _ in range(500):
        data.ctrl[:7] = data.qfrc_bias[:7].copy()
        mujoco.mj_step(model, data)

def draw_path(model, center, traj_func, T_total):
    for i in range(1000):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"path_{i}")
        if sid >= 0:
            Xd, _, _ = traj_func(i*T_total/1000.0, center)
            model.site_pos[sid] = Xd

def hide_trail(model):
    for i in range(1000):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"trail_{i}")
        if sid >= 0:
            model.site_pos[sid] = [0, 0, -10]

def run_sim(ctrl_type, traj_name, T_total=8.0, use_viewer=False, dist_vector=None, dist_is_continuous=False):
    """Run simulation. dist_vector: 7-element array. dist_is_continuous: True for constant torque, False for sinusoidal."""
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    ref_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ref_target")
    ee_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_marker")

    # Removed transparency call to keep arm completely solid

    reset_robot(model, data)
    center = get_ee_position(data, hand_id).copy()
    traj_func = TRAJ_MAP[traj_name]
    draw_path(model, center, traj_func, T_total)
    hide_trail(model)

    ctrl_fn = CTRL_MAP[ctrl_type]

    viewer = None
    if use_viewer:
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.lookat[:] = center
        viewer.cam.distance, viewer.cam.elevation = 0.8, -25

    J_prev = get_jacobian(model, data, hand_id, get_ee_position(data, hand_id))
    N = int(T_total / dt)
    log_t, log_e = [], []
    trail_idx = 0
    t0 = time.time()

    for i in range(N):
        t = i * dt
        Xd, Xd_d, Xd_dd = traj_func(t, center)

        # Update markers
        if ref_id >= 0:
            model.site_pos[ref_id] = Xd
        if ee_id >= 0:
            model.site_pos[ee_id] = get_ee_position(data, hand_id)

        # Drop a trail point more frequently to form a continuous line
        if use_viewer and i % 4 == 0 and trail_idx < 1000:
            tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"trail_{trail_idx}")
            if tid >= 0:
                model.site_pos[tid] = get_ee_position(data, hand_id)
            trail_idx += 1

        tau, J_prev, err = ctrl_fn(model, data, hand_id, Xd, Xd_d, Xd_dd, J_prev, dt)
        # External disturbance torque on selected joints (simulates model uncertainty / payload)
        if dist_vector is not None and t > 1.0:  # apply after initial convergence
            if dist_is_continuous:
                tau[:7] += dist_vector  # Continuous (Constant) Disturbance
            else:
                tau[:7] += dist_vector * np.sin(2*np.pi*1.5*t)  # Discrete (Oscillating) Disturbance
        data.ctrl[:7] = tau
        mujoco.mj_step(model, data)
        log_t.append(t); log_e.append(err)

        if viewer and i % 15 == 0:
            viewer.sync()
            if not viewer.is_running(): break
            wall = time.time() - t0
            if t > wall:
                time.sleep(t - wall)

    if viewer:
        viewer.close()
        time.sleep(1.0)
    return np.array(log_t), np.array(log_e)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════
def plot_single(results, shape, filename):
    plt.figure(figsize=(8, 5))
    for label, (t, e, color, ls) in results.items():
        ax_label = f"{label}  (mean {e.mean()*1000:.2f} mm)"
        plt.plot(t, e*1000, color=color, linestyle=ls, linewidth=1.5, label=ax_label)
    plt.title(f"Tracking Error — {shape.title()} Trajectory", fontsize=14, fontweight="bold")
    plt.xlabel("Time (s)", fontsize=12); plt.ylabel("Error (mm)", fontsize=12)
    plt.legend(fontsize=10); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight"); plt.close()
    print(f"  => Plot saved: {filename}")

def plot_all(all_results, filename="unified_comparison.png"):
    n = len(all_results)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 5*rows))
    if n == 1: axes = [axes]
    else: axes = axes.flat
    fig.suptitle("Controller Tracking Performance Comparison", fontsize=16, fontweight="bold", y=0.98)
    for ax, (shape, res) in zip(axes, all_results.items()):
        for label, (t, e, color, ls) in res.items():
            ax.plot(t, e*1000, color=color, linestyle=ls, linewidth=1.5,
                    label=f"{label} ({e.mean()*1000:.2f}mm)")
        ax.set_title(f"{shape.title()}", fontsize=13)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Error (mm)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    for ax in list(axes)[n:]:
        ax.set_visible(False)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(filename, dpi=200, bbox_inches="tight"); plt.close()
    print(f"\n  => Plot saved: {filename}")

# ══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE MENU
# ══════════════════════════════════════════════════════════════════════════════
def ask_choice(prompt, options):
    for i, opt in enumerate(options, 1):
        print(f"    {i}. {opt}")
    while True:
        val = input(f"  {prompt} [{1}-{len(options)}]: ").strip()
        if val.isdigit() and 1 <= int(val) <= len(options):
            return options[int(val)-1]
        print("    Invalid, try again.")

def ask_disturbance():
    print("\n  Configure External Disturbance Magnitude (Nm):")
    choice = ask_choice("Magnitude", [
        "No Disturbance (0 Nm)",
        "Default Robustness Test (Joints 1-3: 8, 8, 4 Nm)",
        "Custom (Manually enter amplitude for all 7 joints)"
    ])
    
    dist_vector = None
    if choice.startswith("Default"):
        dist_vector = np.array([8.0, 8.0, 4.0, 0.0, 0.0, 0.0, 0.0])
    elif not choice.startswith("No"):
        while True:
            val = input("    Enter 7 numbers separated by spaces (e.g. 10 0 5 0 0 0 0): ").strip()
            val = val.replace("'", "").replace('"', "")
            parts = val.split()
            if len(parts) == 7:
                try:
                    dist_vector = np.array([float(x) for x in parts])
                    break
                except ValueError:
                    print("    Error: Please enter only numbers, not letters.")
            else:
                print(f"    Invalid input: You entered {len(parts)} numbers, but exactly 7 are required.")

    dist_is_continuous = False
    if dist_vector is not None:
        print("\n  Configure Disturbance Type:")
        type_choice = ask_choice("Type", [
            "Constant (A steady, continuous force like a heavy payload)",
            "Oscillating (A time-varying sine wave force)"
        ])
        dist_is_continuous = type_choice.startswith("Constant")

    return dist_vector, dist_is_continuous

if __name__ == "__main__":
    print("\n" + "═"*70)
    print("  FL+PD (Baseline)  vs  FL+SMC (Proposed)  vs  VLA (Gemini+SMC)")
    print("═"*70)

    MODES = ["Run single controller on single shape (with viewer)",
             "Compare all 3 controllers on a chosen shape (with viewer)",
             "Headless: compute all errors for all shapes & generate plots",
             "VLA interactive (Gemini picks the shape)"]

    # Accept mode from CLI: python fl_smc_controller.py 3
    if len(sys.argv) > 1 and sys.argv[1].isdigit() and 1 <= int(sys.argv[1]) <= len(MODES):
        mode_choice = MODES[int(sys.argv[1]) - 1]
        print(f"\n  Auto-selected: {mode_choice}")
    else:
        print("\n  What would you like to do?")
        mode_choice = ask_choice("Select", MODES)
    T = 8.0

    # ── Mode 1: Single controller, single shape, with viewer ──
    if mode_choice == MODES[0]:
        print("\n  Choose controller:")
        ctrl = ask_choice("Controller", ["Baseline (FL+PD)", "Proposed (FL+SMC)"])
        ctrl_key = "PD" if "PD" in ctrl else "SMC"
        print("\n  Choose shape:")
        shape = ask_choice("Shape", ALL_SHAPES)
        
        dist_vector, dist_is_continuous = ask_disturbance()
        
        print(f"\n  Running {ctrl} on {shape} ...")
        t, e = run_sim(ctrl_key, shape, T, use_viewer=True, dist_vector=dist_vector, dist_is_continuous=dist_is_continuous)
        print(f"  Mean error: {e.mean()*1000:.2f} mm")
        fname = f"error_{ctrl_key.lower()}_{shape}.png"
        plot_single(
            {ctrl: (t, e, "crimson" if ctrl_key=="SMC" else "dodgerblue", "-")},
            shape, fname)

    # ── Mode 2: Compare all 3 on one shape, with viewer ──
    elif mode_choice == MODES[1]:
        print("\n  Choose shape:")
        shape = ask_choice("Shape", ALL_SHAPES)
        
        dist_vector, dist_is_continuous = ask_disturbance()
        
        res = {}

        print(f"\n  [1/3] Baseline (FL+PD) on {shape}...")
        t1, e1 = run_sim("PD", shape, T, use_viewer=True, dist_vector=dist_vector, dist_is_continuous=dist_is_continuous)
        res["Baseline (FL+PD)"] = (t1, e1, "dodgerblue", "-")
        print(f"    Mean error: {e1.mean()*1000:.2f} mm")

        print(f"\n  [2/3] Proposed (FL+SMC) on {shape}...")
        t2, e2 = run_sim("SMC", shape, T, use_viewer=True, dist_vector=dist_vector, dist_is_continuous=dist_is_continuous)
        res["Proposed (FL+SMC)"] = (t2, e2, "crimson", "-")
        print(f"    Mean error: {e2.mean()*1000:.2f} mm")

        print(f"\n  [3/3] VLA (Gemini → FL+SMC) on {shape}...")
        vla_shape = vla_choose_trajectory(SHAPE_PROMPTS[shape])
        t3, e3 = run_sim("SMC", vla_shape, T, use_viewer=True, dist_vector=dist_vector, dist_is_continuous=dist_is_continuous)
        res["VLA (Gemini+FL+SMC)"] = (t3, e3, "forestgreen", "--")
        print(f"    Gemini chose: {vla_shape} | Mean error: {e3.mean()*1000:.2f} mm")

        plot_single(res, shape, f"comparison_{shape}.png")

    # ── Mode 3: Headless batch — all shapes, nominal + disturbed ──
    elif mode_choice == MODES[2]:
        dist_arr = np.array([8.0, 8.0, 4.0, 0.0, 0.0, 0.0, 0.0])
        all_nom, all_dist = {}, {}

        for shape in ALL_SHAPES:
            print(f"\n{'─'*50}\n  Shape: {shape.upper()}\n{'─'*50}")

            # --- Nominal (no disturbance) ---
            print(f"  [Nominal] Baseline (FL+PD)...")
            t1, e1 = run_sim("PD", shape, T, use_viewer=False, dist_vector=None)
            print(f"    Mean error: {e1.mean()*1000:.2f} mm")

            print(f"  [Nominal] Proposed (FL+SMC)...")
            t2, e2 = run_sim("SMC", shape, T, use_viewer=False, dist_vector=None)
            print(f"    Mean error: {e2.mean()*1000:.2f} mm")

            all_nom[shape] = {
                "Baseline (FL+PD)":  (t1, e1, "dodgerblue", "-"),
                "Proposed (FL+SMC)": (t2, e2, "crimson", "-"),
            }

            # --- With external disturbance (Report Sec 2.3) ---
            print(f"  [Disturbed Default] Baseline (FL+PD)...")
            t3, e3 = run_sim("PD", shape, T, use_viewer=False, dist_vector=dist_arr)
            print(f"    Mean error: {e3.mean()*1000:.2f} mm")

            print(f"  [Disturbed Default] Proposed (FL+SMC)...")
            t4, e4 = run_sim("SMC", shape, T, use_viewer=False, dist_vector=dist_arr)
            print(f"    Mean error: {e4.mean()*1000:.2f} mm")

            all_dist[shape] = {
                "Baseline (FL+PD)":  (t3, e3, "dodgerblue", "-"),
                "Proposed (FL+SMC)": (t4, e4, "crimson", "-"),
            }

        # Generate nominal comparison
        plot_all(all_nom, "comparison_nominal.png")
        # Generate disturbed comparison
        plot_all(all_dist, "comparison_disturbed.png")

        # Generate combined 2-row figure for report
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        fig.suptitle("Robustness Comparison: Nominal vs External Disturbance",
                     fontsize=16, fontweight="bold")
        for col, shape in enumerate(ALL_SHAPES):
            # Row 0: nominal
            ax = axes[0, col]
            for label, (t, e, c, ls) in all_nom[shape].items():
                ax.plot(t, e*1000, color=c, linestyle=ls, lw=1.5,
                        label=f"{label} ({e.mean()*1000:.2f}mm)")
            ax.set_title(f"{shape.title()} (Nominal)", fontsize=11)
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Error (mm)")
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

            # Row 1: disturbed
            ax = axes[1, col]
            for label, (t, e, c, ls) in all_dist[shape].items():
                ax.plot(t, e*1000, color=c, linestyle=ls, lw=1.5,
                        label=f"{label} ({e.mean()*1000:.2f}mm)")
            ax.set_title(f"{shape.title()} (Disturbed)", fontsize=11)
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Error (mm)")
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0,0,1,0.95])
        plt.savefig("robustness_comparison.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"\n  => Plot saved: robustness_comparison.png")

    # ── Mode 4: VLA interactive ──
    elif mode_choice == MODES[3]:
        cmd = input("\n  Describe what the robot should draw: ")
        shape = vla_choose_trajectory(cmd)
        
        dist_vector, dist_is_continuous = ask_disturbance()
        
        print(f"\n  Gemini chose: {shape}. Running FL+SMC with viewer...")
        t, e = run_sim("SMC", shape, T, use_viewer=True, dist_vector=dist_vector, dist_is_continuous=dist_is_continuous)
        print(f"  Mean error: {e.mean()*1000:.2f} mm")
        plot_single({"VLA (Gemini+FL+SMC)": (t, e, "forestgreen", "-")}, shape, f"vla_{shape}.png")

    print("\n  Done!")

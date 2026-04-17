import mujoco
import mujoco.viewer
import time

# Load scene
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# Approximate actuator settings for reaching different cube positions
TARGET_POSES = {
    "red cube":   [0.0,  0.3,  0.0, -1.2, 0.0, 1.5, 0.8],
    "blue cube":  [-0.4, 0.3,  0.0, -1.2, 0.0, 1.5, 0.8],
    "green cube": [0.4,  0.3,  0.0, -1.2, 0.0, 1.5, 0.8],
}

target_name = "red cube"
target = TARGET_POSES[target_name]

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()

    while viewer.is_running():
        t = time.time() - start

        # Slowly move toward target pose
        for i in range(len(target)):
            data.ctrl[i] = target[i]

        mujoco.mj_step(model, data)
        viewer.sync()

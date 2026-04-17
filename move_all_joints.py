import mujoco
import mujoco.viewer
import math
import time

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    last_actuator = -1

    while viewer.is_running():
        t = time.time() - start

        for i in range(model.nu):
            data.ctrl[i] = 0

        actuator = int(t // 3) % model.nu

        data.ctrl[actuator] = 0.5 * math.sin(2 * t)

        if actuator != last_actuator:
            print("Currently moving actuator", actuator)
            last_actuator = actuator

        mujoco.mj_step(model, data)
        viewer.sync()

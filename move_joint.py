import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()

    while viewer.is_running():
        t = time.time() - start

        # Move first actuator slowly back and forth
        data.ctrl[0] = 0.5 * __import__("math").sin(t)

        mujoco.mj_step(model, data)
        viewer.sync()

import mujoco

model = mujoco.MjModel.from_xml_path("scene.xml")

print("Number of actuators:", model.nu)
print()

for i in range(model.nu):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))

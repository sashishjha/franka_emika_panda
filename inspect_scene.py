import mujoco

model = mujoco.MjModel.from_xml_path("scene.xml")

print("ACTUATORS")
for i in range(model.nu):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))

print("\nJOINTS")
for i in range(model.njnt):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))

print("\nSITES")
for i in range(model.nsite):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i))

print("\nBODIES")
for i in range(model.nbody):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i))

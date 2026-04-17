import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# Body IDs
hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")

# Use first 7 joints of Panda arm
joint_ids = list(range(7))

# Tunable parameters
step_size = 0.4
damping = 1e-2

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        # Update body/world positions
        mujoco.mj_forward(model, data)

        # Current hand position in world coordinates
        hand_pos = data.xpos[hand_id].copy()

        # Cube world position
        cube_pos = data.xpos[cube_id].copy()

        # Desired position: slightly above the cube
        target = cube_pos + np.array([0.0, 0.0, 0.12])

        # Position error
        error = target - hand_pos

        # Stop if already close
        if np.linalg.norm(error) > 0.005:

            # Jacobian of hand body
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))

            mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)

            # Only keep first 7 Panda joints
            J = jacp[:, :7]

            # Damped least-squares IK:
            #
            # dq = J^T (J J^T + λI)^-1 error
            #
            A = J @ J.T + damping * np.eye(3)
            dq = J.T @ np.linalg.solve(A, error)

            # Apply update to joint positions
            for i in range(7):
                data.qpos[i] += step_size * dq[i]

        mujoco.mj_forward(model, data)
        viewer.sync()
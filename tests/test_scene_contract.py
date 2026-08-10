import unittest

import mujoco


class SceneContractTest(unittest.TestCase):
    def test_active_scene_exposes_a_dedicated_policy_camera_and_torque_interface(self):
        model = mujoco.MjModel.from_xml_path("scene_torque.xml")
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "policy_camera")

        self.assertGreaterEqual(camera_id, 0)
        self.assertEqual(model.nu, 8)
        self.assertEqual(model.opt.timestep, 0.002)
        for actuator_id in range(7):
            self.assertEqual(
                model.actuator_trntype[actuator_id],
                mujoco.mjtTrn.mjTRN_JOINT,
            )


if __name__ == "__main__":
    unittest.main()

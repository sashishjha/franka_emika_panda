import unittest

import mujoco

from vla.camera import PolicyCamera


class PolicyCameraTest(unittest.TestCase):
    def test_renderer_returns_rgb_when_a_local_graphics_context_is_available(self):
        model = mujoco.MjModel.from_xml_path("scene_torque.xml")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        try:
            camera = PolicyCamera(model)
        except Exception as exc:
            self.skipTest(f"No local MuJoCo graphics context is available: {exc}")
        try:
            frame = camera.render(data)
        finally:
            camera.close()
        self.assertEqual(frame.shape, (224, 224, 3))
        self.assertEqual(str(frame.dtype), "uint8")


if __name__ == "__main__":
    unittest.main()

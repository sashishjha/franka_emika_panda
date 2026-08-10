import unittest

import numpy as np

from vla.action_adapter import ActionAdapterConfig, ActionValidationError, CartesianActionAdapter


class CartesianActionAdapterTest(unittest.TestCase):
    def test_action_is_clipped_and_converted_to_a_smooth_reference(self):
        adapter = CartesianActionAdapter(ActionAdapterConfig(max_translation_step_m=0.03))
        adapter.reset([0.0, 0.0, 0.0])
        adapter.update_action([1.0, 0.0, 0.0, 0.1, 0.2, 0.3, 1.0])

        self.assertTrue(np.allclose(adapter.target_position, [0.03, 0.0, 0.0]))
        reference = adapter.step(0.002)
        self.assertGreater(reference.position[0], 0.0)
        self.assertLess(reference.position[0], 0.03)
        self.assertEqual(reference.gripper_ctrl, 255.0)
        self.assertTrue(np.allclose(reference.rotation_delta, [0.1, 0.2, 0.3]))

    def test_workspace_and_closed_gripper_are_enforced(self):
        adapter = CartesianActionAdapter(ActionAdapterConfig(
            workspace_lower=(0.0, -0.1, 0.0),
            workspace_upper=(0.02, 0.1, 0.1),
        ))
        adapter.reset([0.01, 0.0, 0.01])
        adapter.update_action([0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.assertTrue(np.allclose(adapter.target_position, [0.02, 0.0, 0.01]))
        self.assertEqual(adapter.step(0.002).gripper_ctrl, 0.0)

    def test_invalid_actions_are_rejected(self):
        adapter = CartesianActionAdapter()
        with self.assertRaises(RuntimeError):
            adapter.step(0.002)
        adapter.reset([0.0, 0.0, 0.0])
        with self.assertRaises(ActionValidationError):
            adapter.update_action([0.0] * 6)
        with self.assertRaises(ActionValidationError):
            adapter.update_action([0.0, 0.0, np.nan, 0.0, 0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()

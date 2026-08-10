import unittest
from unittest.mock import patch

import numpy as np

from fl_smc_controller import run_sim
from vla.policy import ConstantPolicy


class _FakePolicyCamera:
    def __init__(self, *_):
        self.closed = False

    @staticmethod
    def render(_data):
        return np.zeros((224, 224, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


class _ImmediatePolicyRunner:
    """Deterministic stand-in for the asynchronous runner in a fast test loop."""

    def __init__(self, policy):
        self.policy = policy
        self.pending = None
        self.last_error = None

    @property
    def request_in_flight(self):
        return False

    def submit(self, image, instruction):
        self.pending = self.policy.act(image, instruction)
        return True

    def poll(self):
        action, self.pending = self.pending, None
        return action

    def close(self):
        pass


class PolicyLoopTest(unittest.TestCase):
    @patch("fl_smc_controller.AsyncPolicyRunner", _ImmediatePolicyRunner)
    @patch("fl_smc_controller.PolicyCamera", _FakePolicyCamera)
    def test_policy_action_reaches_existing_smc_loop_without_a_trajectory(self):
        time_values, errors = run_sim(
            "SMC",
            T_total=0.04,
            use_viewer=False,
            policy=ConstantPolicy([0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            instruction="move right",
            policy_hz=5.0,
        )

        self.assertEqual(len(time_values), 20)
        self.assertEqual(len(errors), 20)
        self.assertTrue(np.all(np.isfinite(errors)))


if __name__ == "__main__":
    unittest.main()

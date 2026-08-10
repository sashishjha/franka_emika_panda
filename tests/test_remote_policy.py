import base64
import json
import unittest
from unittest.mock import patch

import numpy as np

from vla.policy import RemoteOpenVLAPolicy, RemotePolicyConfig


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    @staticmethod
    def read():
        return json.dumps({"action": [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}).encode()


class RemoteOpenVLAPolicyTest(unittest.TestCase):
    @patch("vla.policy.urlopen", return_value=_Response())
    def test_raw_rgb_protocol_and_action_response(self, mock_urlopen):
        endpoint = "http://policy.example.test/act"
        policy = RemoteOpenVLAPolicy(RemotePolicyConfig(endpoint=endpoint, unnorm_key="panda_sim"))
        image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)

        action = policy.act(image, "move right")

        self.assertTrue(np.allclose(action, [0.01, 0, 0, 0, 0, 0, 1]))
        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["instruction"], "move right")
        self.assertEqual(payload["unnorm_key"], "panda_sim")
        self.assertEqual(payload["image"]["shape"], [4, 5, 3])
        self.assertEqual(base64.b64decode(payload["image"]["data"]), image.tobytes())


if __name__ == "__main__":
    unittest.main()

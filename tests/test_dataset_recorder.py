import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from collect_vla_dataset import EpisodeRecorder


class DatasetRecorderTest(unittest.TestCase):
    def test_recorder_writes_reviewable_policy_only_dataset(self):
        recorder = EpisodeRecorder("Trace a circular path", policy_hz=5.0, max_translation_step_m=0.03)
        recorder(
            t=0.0,
            image=np.zeros((4, 5, 3), dtype=np.uint8),
            desired_position=np.ones(3),
            desired_velocity=np.array([0.1, 0.0, 0.0]),
            desired_acceleration=np.zeros(3),
            actual_position=np.zeros(3),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.npz"
            recorder.save(path, {"trajectory": "circle"})
            with np.load(path) as data:
                self.assertEqual(data["images"].shape, (1, 4, 5, 3))
                self.assertTrue(np.allclose(data["actions"][0], [0.02, 0, 0, 0, 0, 0, 1]))
                metadata = json.loads(str(data["metadata_json"]))
                self.assertEqual(metadata["trajectory"], "circle")
                self.assertEqual(metadata["instruction"], "Trace a circular path")


if __name__ == "__main__":
    unittest.main()

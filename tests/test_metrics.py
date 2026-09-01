import unittest

from align_drift.metrics import accuracy_by_shift


class MetricsTests(unittest.TestCase):
    def test_groups_accuracy_by_shift(self):
        rows = [{"shift": 0, "chosen": "a", "reward_model_choice": "a"}, {"shift": 0, "chosen": "b", "reward_model_choice": "a"}, {"shift": 1, "chosen": "b", "reward_model_choice": "b"}]
        self.assertEqual(accuracy_by_shift(rows, "reward_model_choice"), {"0": .5, "1": 1.0})


if __name__ == "__main__": unittest.main()

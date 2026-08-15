import unittest

import numpy as np

from bump_inverse_design.features import h3_features, single_bump_features


class FeatureTests(unittest.TestCase):
    def test_single_training_and_validation_features(self):
        designs = np.array([[0.0, 0.0], [0.20, 0.04]])
        self.assertEqual(single_bump_features(designs, "raw").shape, (2, 2))
        validation = single_bump_features(designs, "validation")
        self.assertEqual(validation.shape, (2, 4))
        np.testing.assert_allclose(validation[:, 2], [0.0, 0.2])

    def test_h3_training_and_validation_features(self):
        designs = np.array([[5.0, 0.8, 1.1]])
        training = h3_features(designs, "engineered")
        validation = h3_features(designs, "validation")
        self.assertEqual(training.shape, (1, 14))
        self.assertEqual(validation.shape, (1, 7))
        self.assertTrue(np.all(np.isfinite(training)))
        self.assertTrue(np.all(np.isfinite(validation)))


if __name__ == "__main__":
    unittest.main()

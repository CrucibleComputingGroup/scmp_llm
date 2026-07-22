import unittest

import numpy as np

from benchmark.ppl.mp_objectives import objective_errors


class ObjectiveErrorsTest(unittest.TestCase):
    def test_delta_sigma2_matches_definition(self):
        sigma = np.array([
            [0.24, 0.30, 0.50, 0.75],
            [0.02, 0.04, 0.20, 0.45],
        ])
        got = objective_errors(sigma, "delta_sigma2")
        want = np.array([
            [0.0, 0.06 ** 2, 0.26 ** 2, 0.51 ** 2],
            [0.0, 0.02 ** 2, 0.18 ** 2, 0.43 ** 2],
        ])
        np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12)

    def test_delta_sigma2_ignores_irreducible_row_floor(self):
        sigma = np.array([
            [0.24, 0.30, 0.50],
            [0.02, 0.08, 0.28],
        ])
        shifted = sigma + np.array([[5.0], [11.0]])
        np.testing.assert_allclose(
            objective_errors(sigma, "delta_sigma2"),
            objective_errors(shifted, "delta_sigma2"),
            rtol=0.0,
            atol=1e-12,
        )

    def test_delta_sigma2_clamps_nonmonotonic_measurement_noise(self):
        sigma = np.array([[0.20, 0.19, 0.30]])
        got = objective_errors(sigma, "delta_sigma2")
        np.testing.assert_allclose(got, [[0.0, 0.0, 0.01]])

    def test_existing_objectives_are_unchanged(self):
        sigma = np.array([[0.2, 0.4]], dtype=np.float32)
        np.testing.assert_array_equal(objective_errors(sigma, "sigma"), sigma)
        self.assertEqual(objective_errors(sigma, "sigma").dtype, np.float32)
        np.testing.assert_allclose(
            objective_errors(sigma, "sigma2"), sigma ** 2)
        self.assertEqual(objective_errors(sigma, "sigma2").dtype, np.float32)

    def test_rejects_bad_shape_and_name(self):
        with self.assertRaises(ValueError):
            objective_errors(np.array([0.1, 0.2]), "delta_sigma2")
        with self.assertRaises(ValueError):
            objective_errors(np.zeros((1, 2)), "not-an-objective")


if __name__ == "__main__":
    unittest.main()

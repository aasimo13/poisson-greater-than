"""Check the analytic construction against distributions and a separate MILP."""
import pathlib
import sys
import unittest

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from gaussian_theorem import optimal_encoders, required_gap, threshold_errors


class GaussianTheoremTests(unittest.TestCase):
    def test_every_input_pair_and_noise_scale(self):
        for K in (2, 3, 4, 8, 17):
            for epsilon in (0.001, 0.01, 0.1):
                for sigma in (0.25, 2.0):
                    with self.subTest(K=K, epsilon=epsilon, sigma=sigma):
                        x, y, cut = optimal_encoders(K, epsilon, sigma)
                        err = threshold_errors(x, y, cut, sigma)
                        self.assertLessEqual(float(err.max()), epsilon + 2e-14)
                        self.assertAlmostEqual(float(err.max()), epsilon, places=12)
                        self.assertGreaterEqual(min(x.min(), y.min()), 0)

    def test_gap_is_necessary_for_a_critical_pair(self):
        # Shrinking one end of an attaining encoder makes opposite labels
        # closer than the two-hypothesis minimum, regardless of decoder.
        x, y, _ = optimal_encoders(4)
        x[0] -= 0.1 * required_gap()
        false_mean = x[0] + y[0]        # (0,0)
        true_mean = x[1] + y[0]         # (1,0)
        two_point_minimax = norm.cdf(-abs(false_mean - true_mean) / 2)
        self.assertGreater(two_point_minimax, 0.01)

    def test_agrees_with_separate_milp_for_small_alphabet(self):
        from gaussian_milp import solve
        for K in (2, 3):
            with self.subTest(K=K):
                res, _, _, _ = solve(K, time_limit=30)
                self.assertEqual(res.status, 0, res.message)
                x, y, _ = optimal_encoders(K)
                self.assertAlmostEqual(res.fun, x.max() + y.max(), places=6)

    def test_invalid_parameters_are_rejected(self):
        for K in (1, 0, 2.5, True):
            with self.subTest(K=K), self.assertRaises(ValueError):
                optimal_encoders(K)
        for epsilon in (0, 0.5, 1, np.nan, np.inf):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                required_gap(epsilon=epsilon)
        for sigma in (0, -1, np.nan, np.inf):
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                required_gap(sigma=sigma)


if __name__ == "__main__":
    unittest.main()

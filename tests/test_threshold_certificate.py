"""Check the analytic interval omitted by the numerical threshold search."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from threshold_optimum import certify, near_equal_poisson_bound, s_star


class ThresholdCertificateTests(unittest.TestCase):
    def test_prefix_lower_bound_is_conservative_against_actual_tests(self):
        for u in (1.05, 1.2, 1.5):
            with self.subTest(u=u):
                self.assertLess(near_equal_poisson_bound(u), s_star(u))

    def test_default_prefix_is_far_above_feasible_budget(self):
        self.assertGreater(near_equal_poisson_bound(1.0 + 1e-9), 1e18)

    def test_unsafe_start_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "prefix"):
            certify(4, "low", 321.54, 0.0035, u_start=1.5)

    def test_invalid_arguments_are_rejected(self):
        for u, target in ((1.0, 0.01), (1.2, 0.5), (1.2, 0.0)):
            with self.subTest(u=u, target=target), self.assertRaises(ValueError):
                near_equal_poisson_bound(u, target)


if __name__ == "__main__":
    unittest.main()

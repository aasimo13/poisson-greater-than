"""Meaningful checks of the resource accounting and conditional guarantees."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from core import labels, lp_minimax, pmf_matrix
from resource_benchmark import (adaptive_banded, fixed_shared, four_bit_ook,
                                mean_energy, propagate_policy,
                                readout_energy_break_even, shape_rates)


class ResourceBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.band = fixed_shared("banded")
        cls.ook = four_bit_ook()

    def test_known_photon_budget_and_source_rate_cap(self):
        self.assertAlmostEqual(self.band["max_expected_signal_photons"], 196.62, delta=.03)
        for name in ("threshold", "banded"):
            x, y = shape_rates(name, 7.0)
            self.assertLessEqual(max(x.max(), y.max()), 7.0 + 1e-12)
        self.assertLessEqual(max(self.band["error_by_pair"]), .01)

    def test_physical_rate_scaling_preserves_photons_and_errors(self):
        a = fixed_shared("banded", peak_rate=1, background_rate=.01)
        b = fixed_shared("banded", peak_rate=2, background_rate=.02)
        self.assertAlmostEqual(a["active_time_deadline"], 2 * b["active_time_deadline"], delta=1e-5)
        np.testing.assert_allclose(a["expected_signal_photons_by_pair"], b["expected_signal_photons_by_pair"], atol=1e-5)
        np.testing.assert_allclose(a["error_by_pair"], b["error_by_pair"], atol=1e-7)

    def test_stored_background_retuned_shape_is_a_distinct_feasible_control(self):
        nominal = fixed_shared("banded", background_rate=.01, wall_deadline=200)
        retuned = fixed_shared("banded_bg1", background_rate=.01, wall_deadline=200)
        self.assertFalse(nominal["meets_wall_deadline"])
        self.assertTrue(retuned["meets_wall_deadline"])
        self.assertLessEqual(retuned["worst_pair_error"], .01)
        self.assertAlmostEqual(retuned["max_expected_signal_photons"], 323.603, delta=.02)

    def test_ook_uses_comparator_constraint_instead_of_conservative_union_bound(self):
        self.assertLessEqual(self.ook["worst_pair_error"], self.ook["bit_error_union_bound"] + 1e-12)
        self.assertLess(self.ook["worst_pair_error"], .01)
        self.assertGreater(self.ook["worst_pair_error"], .0099)
        self.assertGreater(self.ook["bit_error_union_bound"], .01)
        self.assertLess(self.ook["max_expected_signal_photons"], 4 * np.log(399))
        q = self.ook["symmetric_bit_error_target"]
        self.assertAlmostEqual(self.ook["max_expected_signal_photons"], 4 * np.log((1-q)/q), delta=.0001)
        self.assertEqual(self.ook["max_readouts"], 4)

    def test_same_wall_deadline_excludes_slow_four_readouts(self):
        ook = four_bit_ook(readout_time=50, wall_deadline=200)
        band = fixed_shared("banded", readout_time=50, wall_deadline=200)
        self.assertFalse(ook["meets_wall_deadline"])
        self.assertTrue(band["meets_wall_deadline"])
        impossible = adaptive_banded(readout_time=50, wall_deadline=200, steps=4)
        self.assertFalse(impossible["feasible"])

    def test_forward_propagation_matches_single_terminal_count(self):
        x, y = shape_rates("banded")
        rates = (x[:, None] + y[None, :]).ravel()
        horizon, nmax = 110., 400
        _, d = lp_minimax(rates * horizon, labels(4), return_decoder=True, nmax=nmax)
        policy = np.zeros((4, nmax, 3))
        policy[:, :, 2] = 1
        policy[-1, :, 0] = 1 - d[:nmax]
        policy[-1, :, 1] = d[:nmax]
        policy[-1, :, 2] = 0
        actual = propagate_policy(rates, horizon, policy)
        p = pmf_matrix(rates * horizon, nmax=nmax) @ d
        expected = np.where(labels(4), 1 - p, p)
        np.testing.assert_allclose(actual["errors"], expected, atol=2e-10)
        np.testing.assert_allclose(actual["active_times"], horizon, atol=1e-8)
        np.testing.assert_allclose(actual["checks"], 4, atol=1e-9)

    def test_stopping_lp_has_per_pair_guarantee_and_common_deadline(self):
        adaptive = adaptive_banded(steps=2, readout_time=10, wall_deadline=200)
        self.assertTrue(adaptive["feasible"])
        self.assertLessEqual(max(adaptive["error_by_pair"]), .01)
        self.assertAlmostEqual(adaptive["wall_time_deadline"], 200)
        self.assertLessEqual(max(adaptive["expected_wall_time_by_pair"]), 200 + 1e-8)
        self.assertLess(max(adaptive["tail_error_charged_by_pair"]), 1e-9)
        self.assertAlmostEqual(adaptive["uniform_mean_wall_time"], adaptive["lp_mean_wall_time"], delta=1e-3)

    def test_energy_break_even_consistently_uses_means(self):
        bound = readout_energy_break_even(self.ook, self.band)
        c, power = 2., 3.
        readout = bound["photon_coefficient"] * c + bound["active_time_coefficient"] * power
        self.assertAlmostEqual(mean_energy(self.ook, c, power, readout),
                               mean_energy(self.band, c, power, readout), places=8)


if __name__ == "__main__":
    unittest.main()

"""Checks independent of the Markov implementation's internal bookkeeping."""

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import feedback_sequence as fs


class FeedbackSequenceTests(unittest.TestCase):
    def test_overflow_is_explicit_and_probability_mass_is_preserved(self):
        p, tail, _ = fs.poisson_rows(np.array([0., 2., 25.]), 10)
        np.testing.assert_allclose(p.sum(-1), 1, atol=1e-14)
        np.testing.assert_allclose(p[:, -1], poisson.sf(9, [0., 2., 25.]), atol=1e-14)
        self.assertGreater(tail, .99)

    def test_zero_spillover_has_no_previous_input_dependence(self):
        p, _ = fs.observation_tensor([5, 0], [0, 5], 0)
        for row in p:
            np.testing.assert_allclose(row, p[0], atol=1e-14)
        policies = fs.train_policies(p, fs.pair_labels(2))
        self.assertAlmostEqual(policies["no_feedback"][1].max(),
                               policies["correct_previous_answer"][1].max(), places=8)

    def test_markov_stationary_errors_equal_direct_threshold_probabilities(self):
        x, y, rho = [6., 0.], [0., 6.], .2
        p, meta = fs.observation_tensor(x, y, rho)
        # A deterministic count threshold gives an independent analytic check.
        decoder = np.tile((np.arange(meta["cutoff"] + 1) <= 3).astype(float), (2, 1))
        q = fs.output_probabilities(p, decoder)
        means = (np.array(x)[:, None] + np.array(y)[None, :]).ravel()
        expected = poisson.cdf(3, rho * means[:, None] + means[None, :])
        np.testing.assert_allclose(q[:, :, 0], expected, atol=2e-14)
        m = fs.transition_matrix(q)
        np.testing.assert_allclose(m.sum(1), 1, atol=2e-14)
        w, _, residual = fs.stationary_distribution(m)
        np.testing.assert_allclose(w.reshape(-1, 2).sum(1), .25, atol=2e-14)
        self.assertLess(residual, 1e-13)
        labels = fs.pair_labels(2)
        direct = np.where(labels == 1, 1 - expected.mean(0), expected.mean(0))
        np.testing.assert_allclose(fs.errors_given_state(q, labels, w), direct, atol=2e-14)
        stationary_error = sum(w[2*i + (1-label)] for i, label in enumerate(labels))
        self.assertAlmostEqual(stationary_error, direct.mean(), places=13)

    def test_correct_initialization_matches_oracle_training(self):
        p, _ = fs.observation_tensor([159.665, 11.954, .003, 52.963],
                                     [52.963, .003, 11.984, 159.671], .01)
        labels = fs.pair_labels(4)
        decoder, training_error, _ = fs.train_policies(p, labels)["correct_previous_answer"]
        q = fs.output_probabilities(p, decoder)
        np.testing.assert_allclose(fs.errors_given_state(q, labels, fs.oracle_initial(labels)),
                                   training_error, atol=3e-13)
        m = fs.transition_matrix(q)
        state, _, _ = fs.stationary_distribution(m)
        actual = fs.errors_given_state(q, labels, state)
        # This real case distinguishes actual feedback from a perfect-answer oracle.
        self.assertGreater(np.max(np.abs(actual-training_error)), 1e-6)

    def test_periodic_policy_propagation_matches_known_memory_rule(self):
        # Decoder just copies its previous bit with probability .8 and flips .2.
        q = np.tile(np.array([.2, .8]), (4, 4, 1))
        result = fs.periodic_stress(q, fs.pair_labels(2), [1, 2])
        self.assertAlmostEqual(result["maximum_phase_error"], .5, places=13)
        self.assertAlmostEqual(result["mean_cycle_error"], .5, places=13)

    def test_all_future_bound_covers_long_forward_run_from_wrong_initial_bits(self):
        p, _ = fs.observation_tensor([221.338, 147.583, 73.828, .074],
                                     [.074, 73.828, 147.583, 221.338], .1)
        labels = fs.pair_labels(4)
        d, _, _ = fs.train_policies(p, labels)["correct_previous_answer"]
        q = fs.output_probabilities(p, d)
        m = fs.transition_matrix(q)
        stationary, _, _ = fs.stationary_distribution(m)
        wrong = np.zeros((16, 2))
        wrong[np.arange(16), 1-labels] = 1/16
        bound = fs.startup_future_bound(q, labels, m, wrong.ravel(), stationary, steps=3)
        w = wrong.ravel()
        for _ in range(500):
            self.assertLessEqual(fs.errors_given_state(q, labels, w).max(),
                                 bound["all_subsequent_windows_error_upper_bound"] + 1e-13)
            w = w @ m

    def test_disclosed_history_counterexample(self):
        result = fs.history_counterexample()
        self.assertLess(result["worst_current_pair_error_averaged_uniform_history"], .010001)
        self.assertAlmostEqual(result["fixed_history_error"], .09561117657, places=7)

    def test_invalid_probabilities_are_rejected(self):
        with self.assertRaises(ValueError):
            fs.observation_tensor([1, -1], [0, 1], .1)
        with self.assertRaises(ValueError):
            fs.stationary_distribution(np.array([[1.1, 0], [0, 1]]))


if __name__ == "__main__":
    unittest.main()

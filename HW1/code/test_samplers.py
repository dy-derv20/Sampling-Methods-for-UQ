"""
Unit tests for CAP 6938 Assignment 1: Foundations of Sampling Methods

Run these tests to verify your implementations are correct:
    python test_samplers.py

All tests should pass before submitting your assignment.
"""

import numpy as np
from scipy import stats
import sys

# Import your implementations
from samplers import (
    sample_standard_normal,
    sample_multivariate_gaussian,
    sample_unit_disk,
    sample_d_ball,
    theoretical_ball_acceptance_rate,
    GibbsSampler,
    GibbsMHSampler,
    MetropolisHastings,
)
from distributions import gaussian_log_prob, rosenbrock_log_prob


def test_box_muller():
    """Test Box-Muller produces standard normal samples."""
    print("Testing Box-Muller transform...", end=" ")
    np.random.seed(42)

    n, d = 10000, 3
    samples = sample_standard_normal(n, d)

    # Check shape
    assert samples.shape == (n, d), f"Expected shape ({n}, {d}), got {samples.shape}"

    # Check mean (should be ~0)
    mean = np.mean(samples, axis=0)
    assert np.allclose(mean, 0, atol=0.05), f"Mean should be ~0, got {mean}"

    # Check std (should be ~1)
    std = np.std(samples, axis=0)
    assert np.allclose(std, 1, atol=0.05), f"Std should be ~1, got {std}"

    # Kolmogorov-Smirnov test for normality (per dimension)
    for j in range(d):
        _, p_value = stats.kstest(samples[:, j], 'norm')
        assert p_value > 0.01, f"Dimension {j} failed K-S test (p={p_value:.4f})"

    print("PASSED")


def test_multivariate_gaussian():
    """Test Cholesky-based multivariate Gaussian sampling."""
    print("Testing multivariate Gaussian sampling...", end=" ")
    np.random.seed(42)

    n = 10000
    mu = np.array([1.0, -2.0, 3.0])
    Sigma = np.array([
        [2.0, 0.5, 0.3],
        [0.5, 1.5, 0.2],
        [0.3, 0.2, 1.0]
    ])

    samples = sample_multivariate_gaussian(n, mu, Sigma)

    # Check shape
    assert samples.shape == (n, len(mu)), f"Expected shape ({n}, {len(mu)}), got {samples.shape}"

    # Check mean
    emp_mean = np.mean(samples, axis=0)
    mean_error = np.linalg.norm(emp_mean - mu) / np.linalg.norm(mu)
    assert mean_error < 0.05, f"Mean relative error too large: {mean_error:.4f}"

    # Check covariance
    emp_cov = np.cov(samples.T)
    cov_error = np.linalg.norm(emp_cov - Sigma, 'fro') / np.linalg.norm(Sigma, 'fro')
    assert cov_error < 0.1, f"Covariance relative error too large: {cov_error:.4f}"

    print("PASSED")


def test_unit_disk():
    """Test rejection sampling for unit disk."""
    print("Testing unit disk rejection sampling...", end=" ")
    np.random.seed(42)

    n = 5000
    samples, acc_rate = sample_unit_disk(n, return_acceptance_rate=True)

    # Check shape
    assert samples.shape == (n, 2), f"Expected shape ({n}, 2), got {samples.shape}"

    # Check all samples are inside unit disk
    distances = np.sqrt(np.sum(samples**2, axis=1))
    assert np.all(distances <= 1.0 + 1e-10), "Some samples are outside the unit disk!"

    # Check acceptance rate is close to pi/4
    theoretical_rate = np.pi / 4
    rel_error = abs(acc_rate - theoretical_rate) / theoretical_rate
    assert rel_error < 0.05, f"Acceptance rate {acc_rate:.4f} too far from pi/4={theoretical_rate:.4f}"

    print("PASSED")


def test_d_ball():
    """Test rejection sampling for d-dimensional ball."""
    print("Testing d-ball rejection sampling...", end=" ")
    np.random.seed(42)

    for d in [2, 3, 5]:
        n = 1000
        samples, acc_rate = sample_d_ball(n, d, return_acceptance_rate=True)

        # Check shape
        assert samples.shape == (n, d), f"Expected shape ({n}, {d}), got {samples.shape}"

        # Check all samples are inside unit ball
        distances = np.sqrt(np.sum(samples**2, axis=1))
        assert np.all(distances <= 1.0 + 1e-10), f"d={d}: Some samples outside unit ball!"

        # Check acceptance rate
        theoretical_rate = theoretical_ball_acceptance_rate(d)
        rel_error = abs(acc_rate - theoretical_rate) / theoretical_rate
        assert rel_error < 0.2, f"d={d}: Acceptance rate {acc_rate:.4f} too far from {theoretical_rate:.4f}"

    print("PASSED")


def test_theoretical_acceptance_rate():
    """Test theoretical acceptance rate formula."""
    print("Testing theoretical acceptance rate formula...", end=" ")

    # d=2: should be pi/4
    rate_2d = theoretical_ball_acceptance_rate(2)
    assert np.isclose(rate_2d, np.pi/4, rtol=1e-6), f"d=2: expected pi/4, got {rate_2d}"

    # Rates should decrease with dimension
    rates = [theoretical_ball_acceptance_rate(d) for d in [2, 5, 10, 20]]
    assert all(rates[i] > rates[i+1] for i in range(len(rates)-1)), "Rates should decrease with d"

    # d=20 rate should be very small
    assert rates[-1] < 0.001, f"d=20 rate should be <0.001, got {rates[-1]}"

    print("PASSED")


def test_gibbs_sampler():
    """Test Gibbs sampler for Gaussian target."""
    print("Testing Gibbs sampler...", end=" ")
    np.random.seed(42)

    mu = np.array([1.0, -1.0])
    Sigma = np.array([[1.0, 0.5], [0.5, 1.0]])

    sampler = GibbsSampler(mu, Sigma)
    samples = sampler.sample(n_samples=5000, x0=np.zeros(2), burn_in=1000)

    # Check shape
    assert samples.shape == (5000, 2), f"Expected shape (5000, 2), got {samples.shape}"

    # Check mean
    emp_mean = np.mean(samples, axis=0)
    mean_error = np.linalg.norm(emp_mean - mu)
    assert mean_error < 0.1, f"Mean error too large: {mean_error:.4f}"

    # Check covariance
    emp_cov = np.cov(samples.T)
    cov_error = np.linalg.norm(emp_cov - Sigma, 'fro')
    assert cov_error < 0.2, f"Covariance error too large: {cov_error:.4f}"

    print("PASSED")


def test_metropolis_hastings():
    """Test Metropolis-Hastings sampler for Gaussian target."""
    print("Testing Metropolis-Hastings sampler...", end=" ")
    np.random.seed(42)

    mu = np.array([0.0, 0.0])
    Sigma = np.array([[1.0, 0.0], [0.0, 1.0]])

    def log_prob(x):
        return gaussian_log_prob(x, mu, Sigma)

    sampler = MetropolisHastings(log_prob, d=2, proposal_std=1.0)
    samples, acc_rate = sampler.sample(n_samples=5000, x0=np.zeros(2), burn_in=1000)

    # Check shape
    assert samples.shape == (5000, 2), f"Expected shape (5000, 2), got {samples.shape}"

    # Check acceptance rate is reasonable (not too high or low)
    assert 0.1 < acc_rate < 0.9, f"Acceptance rate {acc_rate:.2f} seems wrong"

    # Check mean is close to true mean
    emp_mean = np.mean(samples, axis=0)
    mean_error = np.linalg.norm(emp_mean - mu)
    assert mean_error < 0.15, f"Mean error too large: {mean_error:.4f}"

    print("PASSED")


def test_gibbs_mh_sampler():
    """Test Gibbs-MH sampler for Gaussian target."""
    print("Testing Gibbs-MH sampler...", end=" ")
    np.random.seed(42)

    mu = np.array([0.0, 0.0])
    Sigma = np.array([[1.0, 0.5], [0.5, 1.0]])

    def log_prob(x):
        return gaussian_log_prob(x, mu, Sigma)

    sampler = GibbsMHSampler(log_prob, d=2, proposal_std=1.0)
    samples, acc_rates = sampler.sample(n_samples=5000, x0=np.zeros(2), burn_in=1000)

    # Check shape
    assert samples.shape == (5000, 2), f"Expected shape (5000, 2), got {samples.shape}"

    # Check acceptance rates are returned per coordinate
    assert len(acc_rates) == 2, f"Expected 2 acceptance rates, got {len(acc_rates)}"

    # Check mean is close to true mean
    emp_mean = np.mean(samples, axis=0)
    mean_error = np.linalg.norm(emp_mean - mu)
    assert mean_error < 0.15, f"Mean error too large: {mean_error:.4f}"

    print("PASSED")


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("Running unit tests for Assignment 1")
    print("=" * 60)

    tests = [
        ("Box-Muller", test_box_muller),
        ("Multivariate Gaussian", test_multivariate_gaussian),
        ("Unit Disk", test_unit_disk),
        ("d-Ball", test_d_ball),
        ("Theoretical Acceptance Rate", test_theoretical_acceptance_rate),
        ("Gibbs Sampler", test_gibbs_sampler),
        ("Metropolis-Hastings", test_metropolis_hastings),
        ("Gibbs-MH Sampler", test_gibbs_mh_sampler),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except NotImplementedError as e:
            print(f"Testing {name}... SKIPPED (not implemented)")
        except AssertionError as e:
            print(f"FAILED")
            failed += 1
            errors.append((name, str(e)))
        except Exception as e:
            print(f"ERROR")
            failed += 1
            errors.append((name, f"Unexpected error: {e}"))

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")

    if errors:
        print("\nFailures:")
        for name, error in errors:
            print(f"  {name}: {error}")

    if failed == 0 and passed > 0:
        print("\nAll implemented tests passed!")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

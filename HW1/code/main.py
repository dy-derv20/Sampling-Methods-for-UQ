#!/usr/bin/env python
"""
CAP 6938 Assignment 1: Foundations of Sampling Methods

Entry point with @handle decorators for each question.
Run with: python main.py <question_number> or python main.py all
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Change to script directory for relative imports
os.chdir(Path(__file__).parent.resolve())

from diagnostics import (
    autocorrelation,
    autocorrelation_plot,
    compute_ess,
    compute_rhat,
    histogram_plot,
    trace_plot,
)
from distributions import (
    gaussian_log_prob,
    rosenbrock_function_2d,
    rosenbrock_log_prob,
    rosenbrock_log_prob_nd,
)
from samplers import (
    GibbsMHSampler,
    GibbsSampler,
    MetropolisHastings,
    sample_d_ball,
    sample_multivariate_gaussian,
    sample_standard_normal,
    sample_unit_disk,
    theoretical_ball_acceptance_rate,
)
from utils import handle, main, savefig, set_seed

# =============================================================================
# Section 1: Gaussian Sampling Basics (15 pts)
# =============================================================================


@handle("1.1")
def q1_1():
    """Implement sample_standard_normal using Box-Muller."""
    set_seed(42)
    print("Testing Box-Muller standard normal sampling...")

    n, d = 10000, 5

    #######################
    # TODO: Implement sample_standard_normal in samplers.py
    # Then this call will work and generate the figure
    #
    # Expected: samples array of shape (10000, 5)
    # Each dimension should have mean ≈ 0, std ≈ 1
    #######################
    samples = sample_standard_normal(n, d)

    print(f"\nGenerated {n} samples of dimension {d}")
    print(f"Sample shape: {samples.shape}")
    print(f"\nEmpirical mean (should be ~0): {np.mean(samples, axis=0)}")
    print(f"\nEmpirical std (should be ~1): {np.std(samples, axis=0)}")

    # Plotting code - do not modify
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(
        samples[:, 0], bins=50, density=True, alpha=0.7, label="Samples"
    )
    x = np.linspace(-4, 4, 100)
    axes[0].plot(
        x,
        np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi),
        "r-",
        lw=2,
        label="True N(0,1)",
    )
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Marginal Distribution (dim 0)")
    axes[0].legend()

    axes[1].scatter(samples[:1000, 0], samples[:1000, 1], alpha=0.3, s=5)
    axes[1].set_xlabel("x_0")
    axes[1].set_ylabel("x_1")
    axes[1].set_title("2D Scatter (first 1000 samples)")
    axes[1].set_aspect("equal")

    plt.tight_layout()
    savefig("q1_1_box_muller.pdf")


@handle("1.2")
def q1_2():
    """Implement sample_multivariate_gaussian using Cholesky."""
    set_seed(42)
    print("Testing multivariate Gaussian sampling via Cholesky...")

    mu = np.array([1.0, -2.0, 3.0])
    Sigma = np.array([[2.0, 0.5, 0.3], [0.5, 1.5, 0.2], [0.3, 0.2, 1.0]])
    n = 10000


    samples = sample_multivariate_gaussian(n, mu, Sigma)

    print(f"\nGenerated {n} samples of dimension {len(mu)}")
    print(f"\nTrue mean: {mu}")
    print(f"Empirical mean: {np.mean(samples, axis=0)}")
    emp_cov = np.cov(samples.T)
    print(f"\nTrue covariance:\n{Sigma}")
    print(f"Empirical covariance:\n{emp_cov}")
    print(f"Frobenius norm of error: {np.linalg.norm(emp_cov - Sigma):.6f}")

    # Plotting code - do not modify
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i in range(3):
        axes[i].hist(
            samples[:, i],
            bins=50,
            density=True,
            alpha=0.7,
            edgecolor="black",
            label="Samples",
        )
        x = np.linspace(
            mu[i] - 4 * np.sqrt(Sigma[i, i]),
            mu[i] + 4 * np.sqrt(Sigma[i, i]),
            100,
        )
        pdf = np.exp(-0.5 * (x - mu[i]) ** 2 / Sigma[i, i]) / np.sqrt(
            2 * np.pi * Sigma[i, i]
        )
        axes[i].plot(
            x, pdf, "r-", lw=2, label=f"True N({mu[i]}, {Sigma[i, i]:.1f})"
        )
        axes[i].set_xlabel(f"$x_{i + 1}$")
        axes[i].set_ylabel("Density")
        axes[i].legend(fontsize=8)
    plt.suptitle(
        "Multivariate Gaussian Sampling via Cholesky Decomposition", fontsize=12
    )
    plt.tight_layout()
    savefig("q1_2_cholesky_gaussian.pdf")


@handle("1.3")
def q1_3():
    """Generate 10K samples from 3D Gaussian, compare stats, create pairplot."""
    set_seed(42)
    print("Generating 10K samples from 3D Gaussian and creating pairplot...")

    mu = np.array([1.0, -2.0, 3.0])
    Sigma = np.array([[2.0, 0.5, 0.3], [0.5, 1.5, 0.2], [0.3, 0.2, 1.0]])
    n = 10000

    #######################
    # This uses your sample_multivariate_gaussian implementation
    #######################

    samples = sample_multivariate_gaussian(n, mu, Sigma)

    emp_mean = np.mean(samples, axis=0)
    emp_cov = np.cov(samples.T)

    print(f"\n{'=' * 50}")
    print("Comparison of True vs Empirical Statistics")
    print(f"{'=' * 50}")
    print(f"\nTrue mean: {mu}")
    print(f"Empirical mean: {emp_mean}")
    print(
        f"Relative error: {np.linalg.norm(emp_mean - mu) / np.linalg.norm(mu) * 100:.4f}%"
    )
    print(
        f"\nRelative Frobenius error: {np.linalg.norm(emp_cov - Sigma) / np.linalg.norm(Sigma) * 100:.4f}%"
    )

    # Plotting code - do not modify
    import pandas as pd

    df = pd.DataFrame(samples, columns=["x_1", "x_2", "x_3"])
    fig = sns.pairplot(df, diag_kind="kde", plot_kws={"alpha": 0.3, "s": 5})
    fig.fig.suptitle("3D Gaussian Pairplot (n=10,000)", y=1.02)
    savefig("q1_3_pairplot.pdf")


# =============================================================================
# Section 2: Monte Carlo Integration (15 pts)
# =============================================================================


@handle("2.1")
def q2_1():
    """MC mean estimation: plot variance vs n (log-log), verify O(1/n) scaling."""
    set_seed(42)
    print("Monte Carlo mean estimation: variance scaling...")

    mu_true = np.array([1.0, -2.0, 3.0])
    Sigma = np.array([[2.0, 0.5, 0.3], [0.5, 1.5, 0.2], [0.3, 0.2, 1.0]])
    n_values = np.logspace(2, 5, 20).astype(int)
    n_trials = 50

    #######################
    # TODO: Compute variance of MC mean estimator for each sample size
    # For each n: run n_trials experiments, compute ||emp_mean - mu_true||^2
    # Average to get variance estimate
    #######################
    variances = []
    for n in n_values:
        errors = []
        for _ in range(n_trials):
            samples = sample_multivariate_gaussian(n, mu_true, Sigma)
            emp_mean = np.mean(samples, axis=0)
            errors.append(np.linalg.norm(emp_mean - mu_true) ** 2)
        variances.append(np.mean(errors))
    variances = np.array(variances)

    # Plotting code - do not modify
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(
        n_values, variances, "o-", label="Empirical variance", markersize=5
    )
    c = variances[0] * n_values[0]
    ax.loglog(n_values, c / n_values, "r--", label=r"$O(1/n)$ reference", lw=2)
    ax.set_xlabel("Number of samples (n)")
    ax.set_ylabel("Mean squared error")
    ax.set_title("MC Mean Estimation: Variance Scaling")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig("q2_1_variance_scaling.pdf")

    slope = np.polyfit(np.log(n_values), np.log(variances), 1)[0]
    print(f"\nLog-log slope: {slope:.3f} (should be ~-1.0 for O(1/n))")


@handle("2.2")
def q2_2():
    """MC covariance estimation: plot Frobenius error vs n."""
    set_seed(42)
    print("Monte Carlo covariance estimation: error scaling...")

    mu = np.array([1.0, -2.0, 3.0])
    Sigma_true = np.array([[2.0, 0.5, 0.3], [0.5, 1.5, 0.2], [0.3, 0.2, 1.0]])
    n_values = np.logspace(2, 5, 20).astype(int)
    n_trials = 30

    mean_errors, std_errors = [], []
    for n in n_values:
        errors = []
        for _ in range(n_trials):
            samples = sample_multivariate_gaussian(n, mu, Sigma_true)
            emp_cov = np.cov(samples.T)
            errors.append(np.linalg.norm(emp_cov - Sigma_true, "fro"))
        mean_errors.append(np.mean(errors))
        std_errors.append(np.std(errors))
    mean_errors, std_errors = np.array(mean_errors), np.array(std_errors)

    # Plotting code
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(
        n_values, mean_errors, "o-", label="Frobenius error", markersize=5
    )
    ax.fill_between(
        n_values, mean_errors - std_errors, mean_errors + std_errors, alpha=0.2
    )
    c = mean_errors[0] * np.sqrt(n_values[0])
    ax.loglog(
        n_values,
        c / np.sqrt(n_values),
        "r--",
        label=r"$O(1/\sqrt{n})$ reference",
        lw=2,
    )
    ax.set_xlabel("Number of samples (n)")
    ax.set_ylabel("Frobenius norm error")
    ax.set_title("MC Covariance Estimation: Error Scaling")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig("q2_2_covariance_error.pdf")


@handle("2.3")
def q2_3():
    """Dimension scaling: fix n=1000, vary d, plot relative error."""
    set_seed(42)
    print("Monte Carlo estimation: dimension scaling...")

    n, dimensions, n_trials = 1000, [2, 5, 10, 20, 50, 100], 30
    mean_rel_errors, cov_rel_errors = [], []

    for d in dimensions:
        A = np.random.randn(d, d)
        Sigma_true = A @ A.T / d + np.eye(d)
        mu_true = np.random.randn(d)
        m_err, c_err = [], []
        for _ in range(n_trials):
            samples = sample_multivariate_gaussian(n, mu_true, Sigma_true)
            m_err.append(
                np.linalg.norm(np.mean(samples, axis=0) - mu_true)
                / np.linalg.norm(mu_true)
            )
            c_err.append(
                np.linalg.norm(np.cov(samples.T) - Sigma_true, "fro")
                / np.linalg.norm(Sigma_true, "fro")
            )
        mean_rel_errors.append(np.mean(m_err))
        cov_rel_errors.append(np.mean(c_err))

    # Plotting code
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        dimensions,
        mean_rel_errors,
        "o-",
        label="Mean relative error",
        markersize=8,
    )
    ax.plot(
        dimensions,
        cov_rel_errors,
        "s-",
        label="Covariance relative error",
        markersize=8,
    )

    # Add expected trend lines
    d_arr = np.array(dimensions)
    # O(1) trend for mean error (fit constant to mean of observed values)
    mean_const = np.mean(mean_rel_errors)
    ax.axhline(
        y=mean_const,
        color="C0",
        linestyle="--",
        alpha=0.7,
        label=r"$O(1)$ trend (mean)",
    )
    # O(sqrt(d)) trend for covariance error (fit to first point)
    cov_coeff = cov_rel_errors[0] / np.sqrt(dimensions[0])
    ax.plot(
        d_arr,
        cov_coeff * np.sqrt(d_arr),
        "C1--",
        alpha=0.7,
        label=r"$O(\sqrt{d})$ trend (cov)",
    )

    ax.set_xlabel("Dimension (d)")
    ax.set_ylabel("Relative error")
    ax.set_title(f"MC Estimation: Dimension Scaling (n={n})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig("q2_3_dimension_scaling.pdf")

    print("\nRelative errors by dimension:")
    for d, me, ce in zip(dimensions, mean_rel_errors, cov_rel_errors):
        print(f"  d={d:3d}: mean error={me:.4f}, cov error={ce:.4f}")


# =============================================================================
# Section 3: Rejection Sampling (15 pts)
# =============================================================================


@handle("3.1")
def q3_1():
    """Sample from unit disk using rejection, plot samples, report acceptance rate."""
    set_seed(42)
    print("Rejection sampling from unit disk...")

    n = 5000

    #######################
    # TODO: Implement sample_unit_disk in samplers.py
    # Expected: samples of shape (5000, 2), acceptance_rate ≈ π/4 ≈ 0.785
    #######################
    samples, acc_rate = sample_unit_disk(n, return_acceptance_rate=True)

    theoretical_rate = np.pi / 4
    print(f"\nSamples generated: {n}")
    print(f"Empirical acceptance rate: {acc_rate:.4f}")
    print(f"Theoretical acceptance rate (π/4): {theoretical_rate:.4f}")

    # Plotting code
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=3)
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "r-", lw=2, label="Unit circle")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"Rejection Sampling: Unit Disk\n(n={n}, acceptance rate={acc_rate:.4f})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig("q3_1_disk_samples.pdf")


@handle("3.2")
def q3_2():
    """Generalize to d-ball, compare empirical vs theoretical acceptance."""
    set_seed(42)
    print("Rejection sampling from d-ball: acceptance rate comparison...")

    dimensions = [2, 3, 5, 10, 15, 20]
    n_proposals = 100000

    #######################
    # TODO: Implement theoretical_ball_acceptance_rate in samplers.py
    # Formula: π^(d/2) / (2^d · Γ(d/2 + 1))
    #######################
    empirical_rates, theoretical_rates = [], []
    for d in dimensions:
        proposals = 2 * np.random.rand(n_proposals, d) - 1
        n_accepted = np.sum(np.sum(proposals**2, axis=1) <= 1)
        empirical_rates.append(n_accepted / n_proposals)
        theoretical_rates.append(theoretical_ball_acceptance_rate(d))
        print(
            f"  d={d:2d}: empirical={empirical_rates[-1]:.6f}, theoretical={theoretical_rates[-1]:.6f}"
        )

    # Plotting code
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(dimensions))
    width = 0.35
    ax.bar(x - width / 2, empirical_rates, width, label="Empirical", alpha=0.8)
    ax.bar(
        x + width / 2, theoretical_rates, width, label="Theoretical", alpha=0.8
    )
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Acceptance rate")
    ax.set_title("Rejection Sampling: Acceptance Rate vs Dimension")
    ax.set_xticks(x)
    ax.set_xticklabels(dimensions)
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    savefig("q3_2_acceptance_rates.pdf")


@handle("3.3")
def q3_3():
    """Plot wasted samples vs dimension, find where method becomes infeasible."""
    set_seed(42)
    print("Rejection sampling: curse of dimensionality analysis...")

    #######################
    # This question demonstrates the curse of dimensionality
    # Uses your theoretical_ball_acceptance_rate implementation
    #######################
    dimensions = list(range(2, 26))
    theoretical_rates = [
        theoretical_ball_acceptance_rate(d) for d in dimensions
    ]
    wasted_fraction = [1 - r for r in theoretical_rates]
    expected_samples = [
        1 / r if r > 0 else float("inf") for r in theoretical_rates
    ]

    threshold = 1e-6
    infeasible_dim = None
    for d, rate in zip(dimensions, theoretical_rates):
        if rate < threshold:
            infeasible_dim = d
            break
    print(
        f"Acceptance rate drops below {threshold} at dimension: {infeasible_dim}"
    )

    # Empirical test
    empirical_dims = [2, 3, 5, 8, 10, 12, 15]
    n_proposals = 10000
    empirical_wasted = []
    for d in empirical_dims:
        proposals = 2 * np.random.rand(n_proposals, d) - 1
        n_accepted = np.sum(np.sqrt(np.sum(proposals**2, axis=1)) <= 1)
        empirical_wasted.append(1 - n_accepted / n_proposals)

    # Plotting code (4-panel figure)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].semilogy(
        dimensions, wasted_fraction, "b-", linewidth=2, label="Theoretical"
    )
    axes[0, 0].scatter(
        empirical_dims,
        empirical_wasted,
        color="red",
        s=100,
        zorder=5,
        label="Empirical",
        marker="x",
    )
    axes[0, 0].axhline(
        y=0.99, color="orange", linestyle="--", alpha=0.7, label="99% wasted"
    )
    axes[0, 0].set_xlabel("Dimension (d)")
    axes[0, 0].set_ylabel("Fraction of Wasted Samples")
    axes[0, 0].set_title("Curse of Dimensionality: Fraction Rejected")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(dimensions, expected_samples, "b-", linewidth=2)
    if infeasible_dim:
        axes[0, 1].axvline(
            x=infeasible_dim,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"d={infeasible_dim}",
        )
    axes[0, 1].axhline(
        y=1e6, color="orange", linestyle="--", alpha=0.7, label="1 million"
    )
    axes[0, 1].set_xlabel("Dimension (d)")
    axes[0, 1].set_ylabel("Expected Proposals per Sample")
    axes[0, 1].set_title("Computational Cost")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 2D visualization
    n_demo = 1000
    proposals_2d = 2 * np.random.rand(n_demo, 2) - 1
    inside_2d = np.sqrt(np.sum(proposals_2d**2, axis=1)) <= 1
    axes[1, 0].scatter(
        proposals_2d[inside_2d, 0],
        proposals_2d[inside_2d, 1],
        c="green",
        s=10,
        alpha=0.6,
        label="Accepted",
    )
    axes[1, 0].scatter(
        proposals_2d[~inside_2d, 0],
        proposals_2d[~inside_2d, 1],
        c="red",
        s=10,
        alpha=0.3,
        label="Rejected",
    )
    theta = np.linspace(0, 2 * np.pi, 100)
    axes[1, 0].plot(np.cos(theta), np.sin(theta), "k-", linewidth=2)
    axes[1, 0].set_aspect("equal")
    axes[1, 0].set_title(f"2D: {100 * np.mean(inside_2d):.1f}% Accepted")
    axes[1, 0].legend()

    bar_dims = [2, 5, 10, 15, 20, 25]
    bar_rates = [theoretical_ball_acceptance_rate(d) * 100 for d in bar_dims]
    axes[1, 1].bar(
        range(len(bar_dims)), bar_rates, color="steelblue", edgecolor="black"
    )
    axes[1, 1].set_xticks(range(len(bar_dims)))
    axes[1, 1].set_xticklabels([f"d={d}" for d in bar_dims])
    axes[1, 1].set_ylabel("Acceptance Rate (%)")
    axes[1, 1].set_title("Acceptance Rate Collapse")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    savefig("q3_3_curse_dimensionality.pdf")


# =============================================================================
# Section 4: MCMC Diagnostics (10 pts)
# =============================================================================


@handle("4.1")
def q4_1():
    """Demonstrate ESS and R-hat diagnostics with examples."""
    set_seed(42)
    print("MCMC Diagnostics: ESS and R-hat Demonstration")
    n = 1000

    # iid samples
    iid_samples = np.random.randn(n)
    ess_iid = compute_ess(iid_samples.reshape(-1, 1))

    # Correlated samples (AR(1) with rho=0.95)
    rho = 0.95
    corr_samples = np.zeros(n)
    corr_samples[0] = np.random.randn()
    for i in range(1, n):
        corr_samples[i] = (
            rho * corr_samples[i - 1] + np.sqrt(1 - rho**2) * np.random.randn()
        )
    ess_corr = compute_ess(corr_samples.reshape(-1, 1))

    # Moderate correlation
    rho2 = 0.5
    mod_samples = np.zeros(n)
    mod_samples[0] = np.random.randn()
    for i in range(1, n):
        mod_samples[i] = (
            rho2 * mod_samples[i - 1] + np.sqrt(1 - rho2**2) * np.random.randn()
        )
    ess_mod = compute_ess(mod_samples.reshape(-1, 1))

    print(f"\nESS Examples (n={n}):")
    print(f"  iid: ESS = {ess_iid:.1f} (ratio: {ess_iid / n:.2f})")
    print(f"  rho=0.5: ESS = {ess_mod:.1f} (ratio: {ess_mod / n:.2f})")
    print(f"  rho=0.95: ESS = {ess_corr:.1f} (ratio: {ess_corr / n:.2f})")

    # R-hat examples
    chain1, chain2, chain3 = (
        np.random.randn(n, 1),
        np.random.randn(n, 1),
        np.random.randn(n, 1),
    )
    rhat_converged = compute_rhat([chain1, chain2, chain3])
    chain1_bad = np.random.randn(n, 1)
    chain2_bad = np.random.randn(n, 1) + 2
    chain3_bad = np.random.randn(n, 1) - 2
    rhat_bad = compute_rhat([chain1_bad, chain2_bad, chain3_bad])
    print(
        f"\nR-hat: Converged = {rhat_converged[0]:.3f}, Non-converged = {rhat_bad[0]:.3f}"
    )

    # Plotting code
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].bar(
        ["iid", "rho=0.5", "rho=0.95"],
        [ess_iid, ess_mod, ess_corr],
        color=["green", "orange", "red"],
    )
    axes[0, 0].axhline(y=n, color="blue", linestyle="--", label=f"n={n}")
    axes[0, 0].set_ylabel("ESS")
    axes[0, 0].set_title("ESS Comparison")
    axes[0, 0].legend()

    axes[0, 1].plot(iid_samples[:200], alpha=0.7, label="iid")
    axes[0, 1].plot(corr_samples[:200], alpha=0.7, label="rho=0.95")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_title("Trace Plots")
    axes[0, 1].legend()

    axes[1, 0].bar(
        ["Converged", "Non-converged"],
        [rhat_converged[0], rhat_bad[0]],
        color=["green", "red"],
    )
    axes[1, 0].axhline(y=1.1, color="orange", linestyle="--", label="Threshold")
    axes[1, 0].set_ylabel("R-hat")
    axes[1, 0].set_title("R-hat Comparison")
    axes[1, 0].legend()

    axes[1, 1].plot(chain1_bad[:100], label="Chain 1", alpha=0.7)
    axes[1, 1].plot(chain2_bad[:100], label="Chain 2", alpha=0.7)
    axes[1, 1].plot(chain3_bad[:100], label="Chain 3", alpha=0.7)
    axes[1, 1].set_title("Non-Converged Chains")
    axes[1, 1].legend()

    plt.tight_layout()
    savefig("q4_1_diagnostics_demo.pdf")


@handle("4.2")
def q4_2():
    """Apply diagnostics to Gaussian samples, verify ESS ≈ n for iid."""
    set_seed(42)
    print("Applying diagnostics to iid Gaussian samples...")

    n, mu = 5000, np.array([1.0, -2.0, 3.0])
    Sigma = np.array([[2.0, 0.5, 0.3], [0.5, 1.5, 0.2], [0.3, 0.2, 1.0]])
    samples = sample_multivariate_gaussian(n, mu, Sigma)

    ess = compute_ess(samples)
    print(f"\nSample size: {n}")
    print(f"ESS: {ess}")
    print(f"ESS/n ratio: {ess / n} (should be ~1.0 for iid)")

    fig, axes = trace_plot(
        samples, labels=["x_1", "x_2", "x_3"], title="Trace Plots: iid Gaussian"
    )
    savefig("q4_2_trace_plots.pdf")


@handle("4.3")
def q4_3():
    """Create 'fake MCMC' with correlation, compare ESS to iid case."""
    set_seed(42)
    print("Comparing iid vs correlated samples...")

    n, rho = 5000, 0.9

    #######################
    # AR(1) process: x[i+1] = rho * x[i] + sqrt(1-rho^2) * z[i]
    # This simulates autocorrelated MCMC output
    #######################
    z = np.random.randn(n)
    x_correlated = np.zeros(n)
    x_correlated[0] = z[0]
    for i in range(1, n):
        x_correlated[i] = rho * x_correlated[i - 1] + np.sqrt(1 - rho**2) * z[i]

    x_iid = np.random.randn(n)
    ess_iid = compute_ess(x_iid.reshape(-1, 1))
    ess_correlated = compute_ess(x_correlated.reshape(-1, 1))
    theoretical_ess = n * (1 - rho) / (1 + rho)

    print(f"\nESS (iid): {ess_iid:.1f}")
    print(f"ESS (correlated): {ess_correlated:.1f}")
    print(f"Theoretical ESS for AR(1): {theoretical_ess:.1f}")

    # Plotting code
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(x_iid[:500], lw=0.5)
    axes[0, 0].set_title(f"iid (ESS={ess_iid:.0f})")
    axes[0, 1].plot(x_correlated[:500], lw=0.5)
    axes[0, 1].set_title(f"Correlated ρ={rho} (ESS={ess_correlated:.0f})")

    acf_iid = autocorrelation(x_iid, 50)
    acf_corr = autocorrelation(x_correlated, 50)
    axes[1, 0].bar(range(51), acf_iid, alpha=0.7)
    axes[1, 0].axhline(y=0.05, color="r", linestyle="--")
    axes[1, 0].set_title("ACF: iid")
    axes[1, 1].bar(range(51), acf_corr, alpha=0.7)
    axes[1, 1].axhline(y=0.05, color="r", linestyle="--")
    axes[1, 1].set_title(f"ACF: ρ={rho}")

    plt.tight_layout()
    savefig("q4_3_correlated_trace.pdf")


# =============================================================================
# Section 5: Gibbs and Metropolis-Hastings Sampling (20 pts)
# =============================================================================


@handle("5.1")
def q5_1():
    """Implement and test Gibbs sampler on correlated 2D Gaussian."""
    set_seed(42)
    print("Gibbs Sampler: Sampling from Correlated 2D Gaussian")

    mu = np.array([1.0, 2.0])
    rho = 0.8
    Sigma = np.array([[1.0, rho], [rho, 1.0]])
    n_samples, burn_in = 5000, 500

    #######################
    # TODO: Implement GibbsSampler class in samplers.py
    # Uses exact conditional sampling for Gaussian targets
    #######################
    sampler = GibbsSampler(mu, Sigma)
    samples = sampler.sample(n_samples, x0=np.zeros(2), burn_in=burn_in)

    emp_mean = np.mean(samples, axis=0)
    emp_corr = np.cov(samples.T)[0, 1] / np.sqrt(
        np.cov(samples.T)[0, 0] * np.cov(samples.T)[1, 1]
    )
    ess = compute_ess(samples)

    print(f"\nTrue mean: {mu}, Empirical: {emp_mean}")
    print(f"True correlation: {rho}, Empirical: {emp_corr:.4f}")
    print(f"ESS: {ess}")

    # Plotting code
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    from scipy.stats import multivariate_normal

    axes[0, 0].scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=5)
    x, y = (
        np.linspace(mu[0] - 3, mu[0] + 3, 100),
        np.linspace(mu[1] - 3, mu[1] + 3, 100),
    )
    X, Y = np.meshgrid(x, y)
    Z = multivariate_normal(mu, Sigma).pdf(np.dstack((X, Y)))
    axes[0, 0].contour(X, Y, Z, levels=5, colors="red", alpha=0.7)
    axes[0, 0].set_title(f"Gibbs Samples (ρ={rho})")
    axes[0, 0].set_aspect("equal")

    axes[0, 1].plot(samples[:500, 0], label="$x_1$", alpha=0.7)
    axes[0, 1].plot(samples[:500, 1], label="$x_2$", alpha=0.7)
    axes[0, 1].set_title("Trace Plot")
    axes[0, 1].legend()

    axes[1, 0].hist(samples[:, 0], bins=50, density=True, alpha=0.7)
    axes[1, 0].axvline(mu[0], color="r", linestyle="--")
    axes[1, 0].set_title("Marginal $x_1$")

    axes[1, 1].hist(samples[:, 1], bins=50, density=True, alpha=0.7)
    axes[1, 1].axvline(mu[1], color="r", linestyle="--")
    axes[1, 1].set_title("Marginal $x_2$")

    plt.suptitle("Gibbs Sampler: Exact Conditional Sampling")
    plt.tight_layout()
    savefig("q5_1_gibbs_sampler.pdf")


@handle("5.2")
def q5_2():
    """Implement and test Metropolis-Hastings sampler, study acceptance rate."""
    set_seed(42)
    print("Metropolis-Hastings: Random Walk Sampler")

    mu = np.array([0.0, 0.0])
    Sigma = np.array([[1.0, 0.5], [0.5, 1.0]])

    def log_prob(x):
        return gaussian_log_prob(x, mu, Sigma)

    #######################
    # TODO: Implement MetropolisHastings class in samplers.py
    # Test with different proposal scales to find optimal acceptance rate
    #######################
    proposal_stds = [0.1, 0.5, 1.0, 2.4, 5.0]
    n_samples, burn_in = 5000, 1000
    results, all_samples = [], {}

    print(f"\n{'Proposal σ':>12} {'Accept Rate':>12} {'ESS':>10}")
    for std in proposal_stds:
        sampler = MetropolisHastings(log_prob, d=2, proposal_std=std)
        samples, acc_rate = sampler.sample(
            n_samples, x0=np.zeros(2), burn_in=burn_in
        )
        ess = np.mean(compute_ess(samples))
        all_samples[std] = samples
        results.append((std, acc_rate, ess))
        print(f"  {std:>10.1f} {acc_rate:>12.3f} {ess:>10.1f}")

    # Plotting code
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for idx, std in enumerate([0.1, 1.0, 5.0]):
        samples = all_samples[std]
        acc_rate = [r[1] for r in results if r[0] == std][0]
        axes[0, idx].scatter(samples[::2, 0], samples[::2, 1], alpha=0.3, s=5)
        axes[0, idx].set_title(f"σ = {std} (acc = {acc_rate:.1%})")
        axes[0, idx].set_aspect("equal")

    stds = [r[0] for r in results]
    axes[1, 0].plot(stds, [r[1] for r in results], "o-")
    axes[1, 0].axhline(
        y=0.234, color="r", linestyle="--", label="Optimal 23.4%"
    )
    axes[1, 0].set_xlabel("Proposal std")
    axes[1, 0].set_ylabel("Acceptance rate")
    axes[1, 0].legend()

    axes[1, 1].plot(stds, [r[2] for r in results], "o-", color="green")
    axes[1, 1].set_xlabel("Proposal std")
    axes[1, 1].set_ylabel("ESS")

    axes[1, 2].plot(all_samples[0.1][:200, 0], label="σ=0.1", alpha=0.7)
    axes[1, 2].plot(all_samples[2.4][:200, 0], label="σ=2.4", alpha=0.7)
    axes[1, 2].set_title("Trace Comparison")
    axes[1, 2].legend()

    plt.suptitle("Metropolis-Hastings: Effect of Proposal Scale")
    plt.tight_layout()
    savefig("q5_2_metropolis_hastings.pdf")


@handle("5.3")
def q5_3():
    """Compare Gibbs vs MH on same target distribution."""
    set_seed(42)
    print("Comparison: Gibbs vs Metropolis-Hastings")

    mu = np.array([1.0, -1.0, 2.0])
    Sigma = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.5], [0.3, 0.5, 1.0]])

    def log_prob(x):
        return gaussian_log_prob(x, mu, Sigma)

    n_samples, burn_in = 5000, 500

    gibbs = GibbsSampler(mu, Sigma)
    gibbs_samples = gibbs.sample(n_samples, x0=np.zeros(3), burn_in=burn_in)
    gibbs_ess = compute_ess(gibbs_samples)

    mh = MetropolisHastings(log_prob, d=3, proposal_std=1.5)
    mh_samples, mh_acc = mh.sample(n_samples, x0=np.zeros(3), burn_in=burn_in)
    mh_ess = compute_ess(mh_samples)

    print(f"\n{'Method':<20} {'ESS (mean)':<15} {'ESS/n':<10}")
    print(
        f"{'Gibbs':<20} {np.mean(gibbs_ess):<15.1f} {np.mean(gibbs_ess) / n_samples:<10.3f}"
    )
    print(
        f"{'MH':<20} {np.mean(mh_ess):<15.1f} {np.mean(mh_ess) / n_samples:<10.3f}"
    )

    # Plotting code
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i in range(3):
        axes[0, i].hist(
            gibbs_samples[:, i], bins=40, density=True, alpha=0.5, label="Gibbs"
        )
        axes[0, i].hist(
            mh_samples[:, i], bins=40, density=True, alpha=0.5, label="MH"
        )
        axes[0, i].axvline(mu[i], color="r", linestyle="--")
        if i == 0:
            axes[0, i].legend()
        axes[0, i].set_title(f"Marginal $x_{i + 1}$")

    axes[1, 0].plot(gibbs_samples[:300, 0], label="Gibbs", alpha=0.7)
    axes[1, 0].plot(mh_samples[:300, 0], label="MH", alpha=0.7)
    axes[1, 0].legend()
    axes[1, 0].set_title("Trace Comparison")

    x_pos = np.arange(3)
    axes[1, 1].bar(x_pos - 0.2, gibbs_ess, 0.4, label="Gibbs")
    axes[1, 1].bar(x_pos + 0.2, mh_ess, 0.4, label="MH")
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(["$x_1$", "$x_2$", "$x_3$"])
    axes[1, 1].set_ylabel("ESS")
    axes[1, 1].legend()

    acf_gibbs = autocorrelation(gibbs_samples[:, 0], 50)
    acf_mh = autocorrelation(mh_samples[:, 0], 50)
    axes[1, 2].plot(acf_gibbs, label="Gibbs")
    axes[1, 2].plot(acf_mh, label="MH")
    axes[1, 2].set_title("Autocorrelation")
    axes[1, 2].legend()

    plt.suptitle("Gibbs vs Metropolis-Hastings Comparison")
    plt.tight_layout()
    savefig("q5_3_gibbs_vs_mh.pdf")


# =============================================================================
# Section 6: Rosenbrock Distribution (15 pts)
# =============================================================================


@handle("6.1")
def q6_1():
    """Implement 2D Rosenbrock log-prob, create contour plot, identify mode."""
    print("2D Rosenbrock Distribution")
    a, b, sigma = 1.0, 100.0, 0.5
    print(f"\nParameters: a={a}, b={b}, σ={sigma}")
    print(f"Mode at: ({a}, {a**2}) = (1, 1)")

    x, y = np.linspace(-2, 3, 200), np.linspace(-1, 5, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.array(
        [
            [
                rosenbrock_log_prob(X[i, j], Y[i, j], a, b, sigma)
                for j in range(200)
            ]
            for i in range(200)
        ]
    )
    Z_prob = np.exp(Z - np.max(Z))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cs = axes[0].contour(X, Y, Z, levels=20)
    axes[0].clabel(cs, inline=True, fontsize=8)
    axes[0].plot(a, a**2, "r*", markersize=15, label=f"Mode ({a}, {a**2})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Log-Probability Contours")
    axes[0].legend()

    cf = axes[1].contourf(X, Y, Z_prob, levels=30, cmap="viridis")
    plt.colorbar(cf, ax=axes[1])
    axes[1].plot(a, a**2, "r*", markersize=15)
    axes[1].set_title("Density (unnormalized)")

    plt.tight_layout()
    savefig("q6_1_rosenbrock_contour.pdf")


@handle("6.2")
def q6_2():
    """Apply Gibbs-MH to Rosenbrock, tune parameters."""
    set_seed(42)
    print("Gibbs-MH on 2D Rosenbrock...")
    a, b, sigma = 1.0, 100.0, 0.5

    def log_prob(x):
        return rosenbrock_log_prob(x[0], x[1], a, b, sigma)

    #######################
    # TODO: Implement GibbsMHSampler class in samplers.py
    # Tune proposal_std for good mixing on Rosenbrock
    #######################
    proposal_std = np.array([0.1, 0.2])
    sampler = GibbsMHSampler(log_prob, d=2, proposal_std=proposal_std)
    samples, acc_rates = sampler.sample(
        20000, x0=np.array([0.0, 0.0]), burn_in=5000
    )

    print(f"\nAcceptance rates: {acc_rates}")
    print(f"ESS: {compute_ess(samples)}")

    # Plotting code
    x_grid, y_grid = np.linspace(-1, 2.5, 100), np.linspace(-0.5, 4, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.array(
        [
            [
                rosenbrock_log_prob(X[i, j], Y[i, j], a, b, sigma)
                for j in range(100)
            ]
            for i in range(100)
        ]
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(
        X, Y, np.exp(Z - np.max(Z)), levels=30, cmap="viridis", alpha=0.6
    )
    ax.scatter(samples[::5, 0], samples[::5, 1], alpha=0.3, s=2, c="red")
    ax.plot(a, a**2, "w*", markersize=15)
    ax.set_title(f"Gibbs-MH on Rosenbrock (acc={np.mean(acc_rates):.2f})")
    savefig("q6_2_rosenbrock_samples.pdf")


@handle("6.3")
def q6_3():
    """Analysis: why Rosenbrock is hard, autocorrelation plots."""
    set_seed(42)
    print("Rosenbrock Analysis: Challenges and Improvements")
    a, b, sigma = 1.0, 100.0, 0.5

    def log_prob(x):
        return rosenbrock_log_prob(x[0], x[1], a, b, sigma)

    sampler = GibbsMHSampler(log_prob, d=2, proposal_std=np.array([0.1, 0.2]))
    samples, _ = sampler.sample(10000, x0=np.array([1.0, 1.0]), burn_in=2000)

    fig, ax = autocorrelation_plot(
        samples,
        max_lag=200,
        labels=["x", "y"],
        title="Rosenbrock: Autocorrelation",
    )
    savefig("q6_3_autocorrelation.pdf")

    print(f"\nESS: {compute_ess(samples)}")


if __name__ == "__main__":
    main()

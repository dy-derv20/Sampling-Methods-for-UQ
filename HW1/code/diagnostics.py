"""
MCMC diagnostics for CAP 6938 Assignment 1: Foundations of Sampling Methods

Contains:
- Effective Sample Size (ESS) computation
- Trace plots
- Autocorrelation plots
- R-hat (potential scale reduction factor)
"""

import numpy as np
import matplotlib.pyplot as plt

# Try to import arviz for advanced diagnostics
try:
    import arviz as az

    HAS_ARVIZ = True
except ImportError:
    HAS_ARVIZ = False
    print("Warning: arviz not installed. Using basic ESS implementation.")


def compute_ess(samples):
    """
    Compute Effective Sample Size (ESS) for MCMC samples.

    ESS measures the number of independent samples equivalent to the
    correlated MCMC samples. For iid samples, ESS ≈ n.

    Parameters
    ----------
    samples : ndarray, shape (n,) or (n, d)
        MCMC samples. If 2D, computes ESS for each dimension.

    Returns
    -------
    float or ndarray
        ESS value(s). If input is 2D, returns array of ESS for each dimension.
    """
    samples = np.atleast_2d(samples)
    if samples.shape[0] == 1:
        samples = samples.T

    n, d = samples.shape

    if HAS_ARVIZ:
        # Use arviz for robust ESS computation
        # arviz expects shape (chains, draws, dimensions)
        data = samples.T.reshape(d, 1, n)  # (d, 1 chain, n draws)
        ess_values = []
        for i in range(d):
            ess = az.ess(data[i : i + 1])
            ess_values.append(float(ess))
        return np.array(ess_values) if d > 1 else ess_values[0]
    else:
        # Basic ESS using autocorrelation
        ess_values = []
        for j in range(d):
            x = samples[:, j]
            ess_values.append(_basic_ess(x))
        return np.array(ess_values) if d > 1 else ess_values[0]


def _basic_ess(x):
    """
    Basic ESS computation using autocorrelation.

    ESS = n / (1 + 2 * sum_{k=1}^{K} rho_k)

    where rho_k is the autocorrelation at lag k and K is chosen
    where autocorrelation becomes negligible.
    """
    n = len(x)
    x = x - np.mean(x)
    var = np.var(x)

    if var == 0:
        return n

    # Compute autocorrelation using FFT
    acf = np.correlate(x, x, mode="full")[n - 1 :]
    acf = acf / (var * np.arange(n, 0, -1))

    # Sum autocorrelations until they become negative or negligible
    # (using the initial monotone sequence estimator)
    rho_sum = 0.0
    for k in range(1, n):
        if acf[k] < 0.05:
            break
        rho_sum += acf[k]

    ess = n / (1 + 2 * rho_sum)
    return max(1, min(ess, n))  # Bound ESS between 1 and n


def compute_rhat(samples_list):
    """
    Compute R-hat (potential scale reduction factor) for multiple chains.

    R-hat measures convergence by comparing within-chain and between-chain
    variance. Values close to 1.0 indicate convergence.

    Parameters
    ----------
    samples_list : list of ndarray
        List of sample arrays, one per chain. Each array has shape (n, d).

    Returns
    -------
    ndarray, shape (d,)
        R-hat values for each dimension
    """
    if HAS_ARVIZ:
        # Stack chains: (n_chains, n_samples, d)
        chains = np.stack(samples_list, axis=0)
        n_chains, n_samples, d = chains.shape

        rhat_values = []
        for j in range(d):
            data = chains[:, :, j : j + 1]  # (chains, draws, 1)
            rhat = az.rhat(data)
            rhat_values.append(float(rhat))
        return np.array(rhat_values)
    else:
        # Basic R-hat implementation
        chains = np.stack(samples_list, axis=0)
        n_chains, n_samples, d = chains.shape

        rhat_values = []
        for j in range(d):
            chain_means = np.mean(chains[:, :, j], axis=1)
            chain_vars = np.var(chains[:, :, j], axis=1, ddof=1)

            # Between-chain variance
            B = n_samples * np.var(chain_means, ddof=1)

            # Within-chain variance
            W = np.mean(chain_vars)

            # Pooled variance estimate
            var_plus = ((n_samples - 1) * W + B) / n_samples

            # R-hat
            rhat = np.sqrt(var_plus / W) if W > 0 else 1.0
            rhat_values.append(rhat)

        return np.array(rhat_values)


def autocorrelation(samples, max_lag=None):
    """
    Compute autocorrelation function for samples.

    Parameters
    ----------
    samples : ndarray, shape (n,) or (n, d)
        Samples to compute autocorrelation for
    max_lag : int, optional
        Maximum lag to compute. Default is n//4.

    Returns
    -------
    ndarray
        Autocorrelation values for lags 0 to max_lag
    """
    samples = np.atleast_1d(samples)
    if samples.ndim == 2:
        samples = samples[:, 0]  # Use first dimension if multi-dimensional

    n = len(samples)
    if max_lag is None:
        max_lag = n // 4

    x = samples - np.mean(samples)
    var = np.var(samples)

    if var == 0:
        return np.ones(max_lag + 1)

    acf = np.correlate(x, x, mode="full")[n - 1 : n + max_lag]
    acf = acf / (var * n)

    return acf


def trace_plot(samples, labels=None, title=None, figsize=(12, 4)):
    """
    Create trace plots for MCMC samples.

    Parameters
    ----------
    samples : ndarray, shape (n, d)
        MCMC samples
    labels : list of str, optional
        Labels for each dimension
    title : str, optional
        Plot title
    figsize : tuple, default=(12, 4)
        Figure size

    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    samples = np.atleast_2d(samples)
    if samples.shape[0] == 1:
        samples = samples.T

    n, d = samples.shape

    if labels is None:
        labels = [f"x_{i + 1}" for i in range(d)]

    fig, axes = plt.subplots(
        d, 1, figsize=(figsize[0], figsize[1] * d), squeeze=False
    )
    axes = axes.flatten()

    for i in range(d):
        axes[i].plot(samples[:, i], lw=0.5, alpha=0.8)
        axes[i].set_ylabel(labels[i])
        axes[i].set_xlabel("Iteration")

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    return fig, axes


def histogram_plot(
    samples, labels=None, true_values=None, title=None, figsize=(12, 4)
):
    """
    Create histogram plots for MCMC samples.

    Parameters
    ----------
    samples : ndarray, shape (n, d)
        MCMC samples
    labels : list of str, optional
        Labels for each dimension
    true_values : list of float, optional
        True parameter values to overlay as vertical lines
    title : str, optional
        Plot title
    figsize : tuple, default=(12, 4)
        Figure size

    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    samples = np.atleast_2d(samples)
    if samples.shape[0] == 1:
        samples = samples.T

    n, d = samples.shape

    if labels is None:
        labels = [f"x_{i + 1}" for i in range(d)]

    ncols = min(d, 4)
    nrows = (d + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(figsize[0], figsize[1] * nrows), squeeze=False
    )
    axes = axes.flatten()

    for i in range(d):
        axes[i].hist(
            samples[:, i], bins=50, density=True, alpha=0.7, edgecolor="black"
        )
        axes[i].set_xlabel(labels[i])
        axes[i].set_ylabel("Density")

        if true_values is not None and i < len(true_values):
            axes[i].axvline(
                true_values[i],
                color="r",
                linestyle="--",
                linewidth=2,
                label="True",
            )
            axes[i].legend()

    # Hide unused axes
    for i in range(d, len(axes)):
        axes[i].set_visible(False)

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    return fig, axes


def autocorrelation_plot(
    samples, max_lag=100, labels=None, title=None, figsize=(10, 4)
):
    """
    Create autocorrelation plots for MCMC samples.

    Parameters
    ----------
    samples : ndarray, shape (n, d)
        MCMC samples
    max_lag : int, default=100
        Maximum lag to plot
    labels : list of str, optional
        Labels for each dimension
    title : str, optional
        Plot title
    figsize : tuple, default=(10, 4)
        Figure size

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    samples = np.atleast_2d(samples)
    if samples.shape[0] == 1:
        samples = samples.T

    n, d = samples.shape

    if labels is None:
        labels = [f"x_{i + 1}" for i in range(d)]

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(d):
        acf = autocorrelation(samples[:, i], max_lag)
        ax.plot(acf, label=labels[i], alpha=0.8)

    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax.axhline(
        y=0.05, color="r", linestyle="--", linewidth=0.5, label="0.05 threshold"
    )
    ax.axhline(y=-0.05, color="r", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.legend()

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return fig, ax

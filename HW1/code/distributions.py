"""
Distribution functions for CAP 6938 Assignment 1: Foundations of Sampling Methods

Contains log-probability functions for:
- Multivariate Gaussian
- 2D Rosenbrock
- N-dimensional Rosenbrock
"""

import numpy as np


def gaussian_log_prob(x, mu, Sigma):
    """
    Compute log-probability of a multivariate Gaussian distribution.

    Parameters
    ----------
    x : ndarray, shape (d,)
        Point at which to evaluate log-prob
    mu : ndarray, shape (d,)
        Mean vector
    Sigma : ndarray, shape (d, d)
        Covariance matrix

    Returns
    -------
    float
        Log-probability at x
    """
    d = len(mu)
    diff = x - mu
    # Use Cholesky for numerical stability
    L = np.linalg.cholesky(Sigma)
    # Solve L @ z = diff for z
    z = np.linalg.solve(L, diff)
    # log|Sigma| = 2 * sum(log(diag(L)))
    log_det = 2 * np.sum(np.log(np.diag(L)))
    # Log probability
    log_prob = -0.5 * (d * np.log(2 * np.pi) + log_det + np.dot(z, z))
    return log_prob


def rosenbrock_log_prob(x, y, a=1.0, b=100.0, sigma=0.5):
    """
    Compute log-probability of 2D Rosenbrock distribution.

    The Rosenbrock function is: f(x,y) = (a-x)^2 + b(y-x^2)^2
    The distribution is: p(x,y) ∝ exp(-f(x,y)/(2σ^2))

    Parameters
    ----------
    x : float
        First coordinate
    y : float
        Second coordinate
    a : float, default=1.0
        Rosenbrock parameter (location of minimum)
    b : float, default=100.0
        Rosenbrock parameter (curvature)
    sigma : float, default=0.5
        Scale parameter

    Returns
    -------
    float
        Unnormalized log-probability at (x, y)
    """
    f = (a - x) ** 2 + b * (y - x**2) ** 2
    return -f / (2 * sigma**2)


def rosenbrock_log_prob_nd(x, a=1.0, b=100.0, sigma=0.5):
    """
    Compute log-probability of N-dimensional Rosenbrock distribution.

    The N-D Rosenbrock function is:
    f(x) = sum_{i=1}^{d-1} [(a - x_i)^2 + b(x_{i+1} - x_i^2)^2]

    The distribution is: p(x) ∝ exp(-f(x)/(2σ^2))

    Parameters
    ----------
    x : ndarray, shape (d,)
        Point in d-dimensional space
    a : float, default=1.0
        Rosenbrock parameter (location of minimum)
    b : float, default=100.0
        Rosenbrock parameter (curvature)
    sigma : float, default=0.5
        Scale parameter

    Returns
    -------
    float
        Unnormalized log-probability at x
    """
    d = len(x)
    f = 0.0
    for i in range(d - 1):
        f += (a - x[i]) ** 2 + b * (x[i + 1] - x[i] ** 2) ** 2
    return -f / (2 * sigma**2)


def rosenbrock_function_2d(x, y, a=1.0, b=100.0):
    """
    Compute the 2D Rosenbrock function value (for contour plots).

    Parameters
    ----------
    x : float or ndarray
        First coordinate(s)
    y : float or ndarray
        Second coordinate(s)
    a : float, default=1.0
        Rosenbrock parameter
    b : float, default=100.0
        Rosenbrock parameter

    Returns
    -------
    float or ndarray
        Function value(s)
    """
    return (a - x) ** 2 + b * (y - x**2) ** 2

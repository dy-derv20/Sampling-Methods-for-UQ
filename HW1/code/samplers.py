"""
Sampling methods for CAP 6938 Assignment 1: Foundations of Sampling Methods

Contains:
- Box-Muller transform for standard normal sampling
- Multivariate Gaussian sampling via Cholesky decomposition
- Rejection sampling for unit disk and d-ball
- Gibbs-MH sampler class
"""
import numpy as np
from numpy import linalg
from scipy.special import gamma


def sample_standard_normal(n, d):
    """
    Sample from standard normal distribution using Box-Muller transform.

    Uses only np.random.rand() for uniform random numbers.

    Parameters
    ----------
    n : int
        Number of samples
    d : int
        Dimension

    Returns
    -------
    ndarray, shape (n, d)
        Standard normal samples
    """

    samples_needed = int(np.ceil((n * d) / 2))
    samples = []
    for i in range(samples_needed):
        U1 = np.random.rand()
        U2 = np.random.rand()
        #Box-Muller Transform (Uniform to Gaussian)
        Z1 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
        Z2 = np.sqrt(-2 * np.log(U1)) * np.sin(2 * np.pi * U2)

        samples.append(Z1)
        samples.append(Z2)

    samples = np.array(samples[:n * d])
    samples = samples.reshape(n,d)
    
    return samples
    


def sample_multivariate_gaussian(n, mu, Sigma):
    """
    Sample from multivariate Gaussian using Cholesky decomposition.

    x = μ + L·z where Σ = L·L^T

    Parameters
    ----------
    n : int
        Number of samples
    mu : ndarray, shape (d,)
        Mean vector
    Sigma : ndarray, shape (d, d)
        Covariance matrix

    Returns
    -------
    ndarray, shape (n, d)
        Gaussian samples
    """
    d = len(mu)
    L = np.linalg.cholesky(Sigma)
    multi_g_samples = []
    g_samples = sample_standard_normal(n , d)

    for z in g_samples:
        x = mu + L @ z
        multi_g_samples.append(x)
    ret = np.array(multi_g_samples)
    print("n = ",  n , " and d = " , d)
    print("Multivariate gaussian samples shape : ", ret.shape)
    return ret

    


def sample_unit_disk(n, return_acceptance_rate=False):
    """
    Sample uniformly from unit disk using rejection sampling.

    Proposal: uniform on [-1, 1]^2
    Accept if x^2 + y^2 <= 1

    Parameters
    ----------
    n : int
        Number of samples desired
    return_acceptance_rate : bool, default=False
        Whether to return acceptance rate

    Returns
    -------
    samples : ndarray, shape (n, 2)
        Samples from unit disk
    acceptance_rate : float (only if return_acceptance_rate=True)
        Empirical acceptance rate
    """
    samples = []
    total_proposed = 0
    total_accepted = 0

    while len(samples) < n:
        # Propose from [-1, 1]^2
        batch_size = max(n - len(samples), 1000)

        proposals = np.random.uniform(-1,1,size=(batch_size,2))
        total_proposed += batch_size
        x = proposals[:,0]
        y = proposals[:,1]
        inside_circle = (x**2 + y**2 <= 1)

        accepted = proposals[inside_circle]
        total_accepted += accepted.shape[0]
        samples.extend(accepted.tolist())

    samples = np.array(samples[:n])
    acceptance_rate = total_accepted / total_proposed

    if return_acceptance_rate:
        return samples, acceptance_rate
    return samples


def sample_d_ball(n, d, return_acceptance_rate=False):
    """
    Sample uniformly from d-dimensional unit ball using rejection sampling.

    Proposal: uniform on [-1, 1]^d
    Accept if ||x||^2 <= 1

    Theoretical acceptance rate: V(B_d) / V([-1,1]^d) = π^(d/2) / (2^d · Γ(d/2 + 1))

    Parameters
    ----------
    n : int
        Number of samples desired
    d : int
        Dimension
    return_acceptance_rate : bool, default=False
        Whether to return acceptance rate

    Returns
    -------
    samples : ndarray, shape (n, d)
        Samples from d-ball
    acceptance_rate : float (only if return_acceptance_rate=True)
        Empirical acceptance rate
    """
    samples = []
    total_proposed = 0
    total_accepted = 0

    while len(samples) < n:
        # Propose from [-1, 1]^d (use larger batches for higher d)
        batch_size = max((n - len(samples)) * 10, 1000)

        proposals = np.random.uniform(-1 , 1 , size=(batch_size , d))
        total_proposed += batch_size
        
        inside_ball = np.sum(proposals**2, axis=1) <= 1

        accepted = proposals[inside_ball]
        total_accepted += accepted.shape[0]
        samples.extend(accepted.tolist())
        

    samples = np.array(samples[:n])
    acceptance_rate = total_accepted / total_proposed

    if return_acceptance_rate:
        return samples, acceptance_rate
    return samples


def theoretical_ball_acceptance_rate(d):
    """
    Compute theoretical acceptance rate for d-ball rejection sampling.

    Rate = V(B_d) / V([-1,1]^d) = π^(d/2) / (2^d · Γ(d/2 + 1))

    Parameters
    ----------
    d : int
        Dimension

    Returns
    -------
    float
        Theoretical acceptance rate
    """

    acceptance_rate = np.pi**(d/2) / (2**d * gamma(d/2 + 1))
    return acceptance_rate
    
    


class GibbsMHSampler:
    """
    Gibbs sampler with Metropolis-Hastings steps for each coordinate.

    Uses coordinate-wise updates with Gaussian proposals.

    Parameters
    ----------
    log_prob_fn : callable
        Function that computes log-probability. Should take a 1D array
        of shape (d,) and return a scalar.
    d : int
        Dimension of the target distribution
    proposal_std : float or ndarray, default=1.0
        Standard deviation(s) for Gaussian proposal. If scalar, same std
        is used for all coordinates. If array of shape (d,), coordinate-wise stds.
    """

    def __init__(self, log_prob_fn, d, proposal_std=1.0):
        self.log_prob_fn = log_prob_fn
        self.d = d
        if np.isscalar(proposal_std):
            self.proposal_std = np.full(d, proposal_std)
        else:
            self.proposal_std = np.array(proposal_std)

    def sample(self, n_samples, x0, burn_in=1000):
        """
        Generate samples using Gibbs-MH algorithm.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate (after burn-in)
        x0 : ndarray, shape (d,)
            Initial point
        burn_in : int, default=1000
            Number of burn-in iterations to discard

        Returns
        -------
        samples : ndarray, shape (n_samples, d)
            Generated samples
        acceptance_rates : ndarray, shape (d,)
            Acceptance rate for each coordinate
        """
        x = np.array(x0, dtype=float)
        samples = np.zeros((n_samples, self.d))

        # Track acceptances for each coordinate
        acceptances = np.zeros(self.d)
        total_proposals = np.zeros(self.d)

        total_iterations = n_samples + burn_in
        current_log_prob = self.log_prob_fn(x)

        for i in range(total_iterations):
            # Cycle through each coordinate
            for j in range(self.d):
                #######################
                # TODO: Implement coordinate-wise Metropolis-Hastings step
                #
                # For coordinate j:
                # 1. Propose a new value by perturbing x[j] with Gaussian noise
                # 2. Compute the log acceptance ratio (proposed - current)
                # 3. Accept/reject using the Metropolis criterion
                # 4. Track acceptances for diagnostics (only after burn-in)
                #######################
                raise NotImplementedError("Implement Gibbs-MH coordinate update")

            # Store sample after burn-in
            if i >= burn_in:
                samples[i - burn_in] = x.copy()

        acceptance_rates = acceptances / total_proposals

        return samples, acceptance_rates


class GibbsSampler:
    """
    Gibbs sampler for multivariate Gaussian distributions.

    For Gaussian targets, the conditional distributions are known analytically,
    so we can sample exactly from each conditional (no MH step needed).

    Parameters
    ----------
    mu : ndarray, shape (d,)
        Mean of the target Gaussian
    Sigma : ndarray, shape (d, d)
        Covariance matrix of the target Gaussian
    """

    def __init__(self, mu, Sigma):
        self.mu = np.array(mu)
        self.Sigma = np.array(Sigma)
        self.d = len(mu)

        # Precompute conditional parameters for each coordinate
        # For Gaussian: p(x_j | x_{-j}) is Gaussian with known mean and variance
        self._precompute_conditionals()

    def _precompute_conditionals(self):
        """Precompute conditional distribution parameters."""
        self.cond_var = np.zeros(self.d)
        self.cond_coeffs = []

        for j in range(self.d):
            #######################
            # TODO: Precompute conditional parameters for coordinate j
            #
            # For a Gaussian, p(x_j | x_{-j}) is also Gaussian with:
            # - Conditional variance: Var(x_j | x_{-j}) (store in self.cond_var[j])
            # - Coefficients for conditional mean (store in self.cond_coeffs)
            #
            # Use the formulas from the assignment PDF for conditional Gaussians.
            # Hint: You'll need to extract submatrices from self.Sigma
            #######################
            raise NotImplementedError("Implement _precompute_conditionals")

    def _conditional_mean(self, j, x):
        """Compute conditional mean of x_j given x_{-j}."""
        idx = [i for i in range(self.d) if i != j]
        x_rest = x[idx]
        mu_rest = self.mu[idx]
        return self.mu[j] + self.cond_coeffs[j] @ (x_rest - mu_rest)

    def sample(self, n_samples, x0, burn_in=1000):
        """
        Generate samples using Gibbs sampling.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate (after burn-in)
        x0 : ndarray, shape (d,)
            Initial point
        burn_in : int, default=1000
            Number of burn-in iterations to discard

        Returns
        -------
        samples : ndarray, shape (n_samples, d)
            Generated samples
        """
        x = np.array(x0, dtype=float)
        samples = np.zeros((n_samples, self.d))

        total_iterations = n_samples + burn_in

        for i in range(total_iterations):
            # Cycle through each coordinate
            for j in range(self.d):
                #######################
                # TODO: Sample from conditional distribution
                #
                # Sample x[j] from its conditional distribution given x_{-j}
                # Use self._conditional_mean(j, x) and self.cond_var[j]
                #######################
                raise NotImplementedError("Implement Gibbs coordinate sampling")

            # Store sample after burn-in
            if i >= burn_in:
                samples[i - burn_in] = x.copy()

        return samples


class MetropolisHastings:
    """
    Random Walk Metropolis-Hastings sampler.

    Proposes new states by adding Gaussian noise to current state.

    Parameters
    ----------
    log_prob_fn : callable
        Function that computes log-probability. Should take a 1D array
        of shape (d,) and return a scalar.
    d : int
        Dimension of the target distribution
    proposal_std : float, default=1.0
        Standard deviation for Gaussian proposal (isotropic)
    """

    def __init__(self, log_prob_fn, d, proposal_std=1.0):
        self.log_prob_fn = log_prob_fn
        self.d = d
        self.proposal_std = proposal_std

    def sample(self, n_samples, x0, burn_in=1000):
        """
        Generate samples using Metropolis-Hastings algorithm.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate (after burn-in)
        x0 : ndarray, shape (d,)
            Initial point
        burn_in : int, default=1000
            Number of burn-in iterations to discard

        Returns
        -------
        samples : ndarray, shape (n_samples, d)
            Generated samples
        acceptance_rate : float
            Overall acceptance rate
        """
        x = np.array(x0, dtype=float)
        samples = np.zeros((n_samples, self.d))

        acceptances = 0
        total_proposals = 0

        total_iterations = n_samples + burn_in
        current_log_prob = self.log_prob_fn(x)

        for i in range(total_iterations):
            #######################
            # TODO: Implement Metropolis-Hastings step
            #
            # 1. Propose a new state by adding Gaussian noise to current state
            # 2. Compute the log acceptance ratio
            # 3. Accept/reject using the Metropolis criterion
            # 4. Store sample and track acceptance (only after burn-in)
            #######################
            raise NotImplementedError("Implement Metropolis-Hastings step")

        acceptance_rate = acceptances / total_proposals

        return samples, acceptance_rate

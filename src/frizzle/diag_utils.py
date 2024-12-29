
""" Approximate methods to estimate the diagonal of large linear operators. """

import numpy as np
from typing import Callable, Union, Tuple
from scipy.linalg import eigh_tridiagonal

# Note to self: Hutchinson seems to be the best method for our purposes.

def hutchinson_diagonal_estimator(
    matvec: Callable,
    shape: Tuple[int, int],
    num_samples: int = 100,
    rademacher: bool = True
) -> np.ndarray:
    """
    Estimate diagonal elements using Hutchinson's estimator with
    Rademacher or Gaussian random vectors.
    
    Parameters:
    -----------
    matvec : Callable
        Function that implements the matrix-vector product Ax
    shape : tuple
        Shape of the operator (m, n)
    num_samples : int
        Number of random vectors to use
    rademacher : bool
        If True, use Rademacher distribution (±1), else use Gaussian
        
    Returns:
    --------
    diag_est : ndarray
        Estimated diagonal elements of the operator
    """
    m, n = shape
    diag_est = np.zeros(n)
    
    for _ in range(num_samples):
        # Generate random vector
        if rademacher:
            v = np.random.choice([-1, 1], size=n)
        else:
            v = np.random.randn(n)
            
        # Compute matrix-vector product
        Av = matvec(v)
        
        # Update estimate
        diag_est += (v * Av) / num_samples
        
    return diag_est


def stochastic_lanczos_diag(
    matvec: Callable,
    rmatvec: Callable,
    shape: Tuple[int, int],
    num_samples: int = 10,
    lanczos_steps: int = 30,
    non_negative: bool = True,
    seed: int = None
) -> np.ndarray:
    """
    Estimate diagonal elements of a linear operator using the Stochastic Lanczos method.
    
    Parameters:
    -----------
    matvec : Callable
        Function that implements the matrix-vector product Ax
    rmatvec : Callable
        Function that implements the matrix-vector product A^T x
    shape : tuple
        Shape of the operator (m, n)
    num_samples : int, optional
        Number of random vectors to use for estimation
    lanczos_steps : int, optional
        Number of Lanczos iterations for each random vector
    non_negative : bool, optional
        If True, enforce non-negativity constraint on diagonal estimates
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    diag_est : ndarray
        Estimated diagonal elements of the operator
    """
    if seed is not None:
        np.random.seed(seed)
    
    m, n = shape
    diag_est = np.zeros(n)
    running_variance = np.zeros(n)  # Track variance for each diagonal element
    
    for sample in range(num_samples):
        # Generate random vector
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)
        
        # Initialize arrays for Lanczos iteration
        alpha = np.zeros(lanczos_steps)
        beta = np.zeros(lanczos_steps - 1)
        V = np.zeros((n, lanczos_steps))
        V[:, 0] = v
        
        # Lanczos iteration
        w = matvec(v)
        alpha[0] = np.dot(v, w)
        w = w - alpha[0] * v
        
        for j in range(lanczos_steps - 1):
            beta[j] = np.linalg.norm(w)
            if beta[j] < 1e-12:
                lanczos_steps = j + 1
                alpha = alpha[:lanczos_steps]
                beta = beta[:lanczos_steps-1]
                V = V[:, :lanczos_steps]
                break
                
            V[:, j+1] = w / beta[j]
            v = V[:, j+1]
            
            w = matvec(v)
            w = w - beta[j] * V[:, j]
            alpha[j+1] = np.dot(v, w)
            w = w - alpha[j+1] * v
        
        # Compute eigendecomposition of tridiagonal matrix
        eigvals, eigvecs = eigh_tridiagonal(alpha, beta)
        
        # Update diagonal estimate for each position
        sample_estimates = np.zeros(n)
        for i in range(n):
            temp_est = 0
            for k in range(lanczos_steps):
                v_ik = V[i, k]
                temp_est += (v_ik ** 2) * eigvals[k]
            sample_estimates[i] = temp_est
        
        # Update running statistics
        if sample == 0:
            diag_est = sample_estimates
        else:
            old_mean = diag_est.copy()
            diag_est = diag_est + (sample_estimates - diag_est) / (sample + 1)
            running_variance = (running_variance * sample + 
                              (sample_estimates - old_mean) * 
                              (sample_estimates - diag_est)) / (sample + 1)
    
    # Apply non-negativity constraint if requested
    if non_negative:
        # Project to non-negative orthant
        diag_est = np.maximum(diag_est, 0)
        
        # Compute confidence intervals (95%)
        std_error = np.sqrt(running_variance / num_samples)
        confidence_interval = 1.96 * std_error
        
        # Use more conservative estimate for elements close to zero
        near_zero_mask = diag_est < confidence_interval
        diag_est[near_zero_mask] = 0
    
    return diag_est

def unit_vector_diagonal_estimator(
    matvec: Callable,
    shape: Tuple[int, int],
    num_samples: int = None,
    indices: np.ndarray = None,
    batch_size: int = 1000
) -> np.ndarray:
    """
    Estimate diagonal elements of a linear operator using unit vectors.
    This is a deterministic approach when num_samples is None.
    
    Parameters:
    -----------
    matvec : Callable
        Function that implements the matrix-vector product Ax
    shape : tuple
        Shape of the operator (m, n)
    num_samples : int, optional
        If provided, randomly sample this many diagonal elements
    indices : ndarray, optional
        Specific indices to compute (if None, compute all or num_samples)
    batch_size : int
        Number of elements to compute in each batch
        
    Returns:
    --------
    diag_est : ndarray
        Estimated diagonal elements of the operator
    """
    m, n = shape
    
    if indices is None:
        if num_samples is None:
            indices = np.arange(n)
        else:
            indices = np.random.choice(n, size=num_samples, replace=False)
    
    diag_est = np.zeros(n)
    
    # Process in batches to avoid memory issues
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        # Create batch of unit vectors
        V = np.zeros((n, len(batch_indices)))
        V[batch_indices, np.arange(len(batch_indices))] = 1.0
        
        # Compute matvec for all vectors in batch
        AV = np.zeros((m, len(batch_indices)))
        for j, v in enumerate(V.T):
            AV[:, j] = matvec(v)
        
        # Extract diagonal elements
        diag_est[batch_indices] = np.diagonal(V.T @ AV)
    
    return diag_est

from time import time
import jax
import jax.numpy as jnp
import lineax as lx
import jaxopt
import numpy as np
from functools import partial
from sklearn.neighbors import KDTree
from typing import Optional

from jax_finufft import nufft1, nufft2
from .utils import check_inputs, combine_flags

jax.config.update("jax_enable_x64", True)

def frizzle(
    λ_out: jnp.array,
    λ: jnp.array,
    flux: jnp.array,
    ivar: Optional[jnp.array] = None,
    mask: Optional[jnp.array] = None,
    flags: Optional[jnp.array] = None,
    censor_missing_regions: Optional[bool] = True,
    n_modes: Optional[int] = None,
    rtol: Optional[float] = 1e-6,
    atol: Optional[float] = 1e-6,
    rcond: Optional[float] = None,
):
    """
    Combine spectra by forward modeling.

    :param λ_out:
        The wavelengths to sample the combined spectrum on.
    
    :param λ:
        The wavelengths of the individual spectra. This should be shape (N, ) where N is the number of pixels.
    
    :param flux:
        The flux values of the individual spectra. This should be shape (N, ).
    
    :param ivar: [optional]
        The inverse variance of the individual spectra. This should be shape (N, ).
    
    :param mask: [optional]
        The mask of the individual spectra. If given, this should be a boolean array (pixels with `True` get ignored) of shape (N, ).
        The mask is used to ignore pixel flux when combining spectra, but the mask is not used when computing combined pixel flags.
    
    :param flags: [optional]
        An optional integer array of bitwise flags. If given, this should be shape (N, ).
        
    :param censor_missing_regions: [optional]
        If `True`, then regions where there is no data will be set to NaN in the combined spectrum. If `False` the values evaluated
        from the model will be reported (and have correspondingly large uncertainties) but this will produce unphysical features.

    :param n_modes: [optional]
        The number of Fourier modes to use. If `None` is given then this will default to `len(λ_out)`.
            
    :param rtol: [optional]
        The relative tolerance for the linear solver.
    
    :param atol: [optional]
        The absolute tolerance for the linear solver.
    
    :param rcond: [optional]
        Cut-off ratio for small singular values of the linear system. If `None` (default) the optimal value will be used
        to reduce floating point errors.
            
    :returns:
        A four-length tuple of ``(flux, ivar, flags, meta)`` where:
            - ``flux`` is the combined fluxes,
            - ``ivar`` is the variance of the combined fluxes,
            - ``flags`` are the combined flags, and 
            - ``meta`` is a dictionary.
    """

    λ_out, λ, flux, ivar, mask = check_inputs(λ_out, λ, flux, ivar, mask)

    y_star, C_inv_star, meta = _frizzle(
        λ_out, λ[~mask], flux[~mask], ivar[~mask], 
        n_modes or min(len(λ_out), jnp.sum(~mask)), 
    )
    
    if censor_missing_regions:
        # Set NaNs for regions where there were NO data.
        # Here we check to see if the closest input value was more than the output pixel width.
        tree = KDTree(λ.reshape((-1, 1)))
        distances, indices = tree.query(λ_out.reshape((-1, 1)), k=1)

        no_data = jnp.hstack([distances[:-1, 0] > jnp.diff(λ_out), False])
        meta["no_data_mask"] = no_data
        if jnp.any(no_data):
            y_star = jnp.where(no_data, jnp.nan, y_star)
            C_inv_star = jnp.where(no_data, 0, C_inv_star)

    flags_star = combine_flags(λ_out, λ, flags)

    return (y_star, C_inv_star, flags_star, meta)






def matvec(c, x, w, n_modes):
    return jnp.real(nufft2(_pre_matvec(c, n_modes), x)) * w

def rmatvec(f, x, w, n_modes):
    return _post_rmatvec(nufft1(n_modes, f.astype(jnp.complex128) * w, x), n_modes)

def _ATCinvAv(c, x, C_inv_sqrt, n_modes):
    return rmatvec(matvec(c, x, C_inv_sqrt, n_modes), x, C_inv_sqrt, n_modes)

def ATCinvA_diag(x, C_inv_sqrt, n_modes):
    """
    Compute the diagonal of the ATCinvA matrix.
    
    :param x:
        The x-coordinates of the data.
    
    :param C_inv_sqrt:
        The square root of the inverse variance of the data.
    
    :param n_modes:
        The number of Fourier modes.
    
    :returns:
        The diagonal of the ATCinvA matrix.
    """
    f = jax.jit(partial(_ATCinvAv, x=x, C_inv_sqrt=C_inv_sqrt, n_modes=n_modes))
    return jax.vmap(f, in_axes=(0, ))(jnp.eye(n_modes))


matmat = jax.vmap(matvec, in_axes=(0, None, None, None))
rmatmat = jax.vmap(rmatvec, in_axes=(0, None, None, None))




def nll(θ, flux, ivar, mv):
    return 0.5 * jnp.sum(ivar * (mv(θ) - flux)**2)

hess = jax.hessian(nll)

def _ATCinvAv_efficient(c, x, C_inv_sqrt, n_modes):
    return rmatvec(matvec(c, x, C_inv_sqrt, n_modes), x, C_inv_sqrt, n_modes)

def compute_diagonal_efficient(x, x_star, n_modes, C_inv_diag):

    def B_apply(c):
        return rmatvec(C_inv_diag * matvec(c, x, 1, n_modes), x, 1, n_modes)
    
    def compute_single_diagonal(e_i):
        # Get i-th row of A_star
        #e_i = jnp.zeros(n_modes)  # A_star is (n × m)
        #e_i = e_i.at[i].set(1.0)
        
        # This gives us A_star[i, :] (i-th row)
        a_star_i_row = rmatvec(e_i, x_star, 1, n_modes)  # This is correct!
        
        # Solve B @ y = a_star_i_row  
        y_i = jax.scipy.sparse.linalg.cg(B_apply, a_star_i_row, maxiter=100)[0]
        
        # Diagonal element = a_star_i_row @ y_i
        return jnp.dot(a_star_i_row, y_i)
    
    diagonal = jax.vmap(compute_single_diagonal)(jnp.eye(n_modes))
    return diagonal
    

def ATCinvAv(v, x, C_inv_sqrt, n_modes):
    return rmatmat(matmat(v, x, C_inv_sqrt, n_modes), x, C_inv_sqrt, n_modes)


@partial(jax.jit, static_argnames=("n_modes", ))
def _frizzle(λ_out, λ, flux, ivar, n_modes):

    λ_all = jnp.hstack([λ, λ_out])
    λ_min, λ_max = (jnp.min(λ_all), jnp.max(λ_all))

    small = (λ_max - λ_min)/(1 + len(λ_out))
    scale = (1 - small) * 2 * jnp.pi / (λ_max - λ_min)
    x = (λ - λ_min) * scale
    x_star = (λ_out - λ_min) * scale

    C_inv_sqrt = jnp.sqrt(ivar)
    
    x_star_mv = lambda c: matvec(c, x_star, 1, n_modes)
    weighted_x_mv = lambda c: matvec(c, x, C_inv_sqrt, n_modes)
    θ = jaxopt.linear_solve.solve_normal_cg(
        weighted_x_mv,
        flux * C_inv_sqrt, 
        init=jnp.zeros(n_modes)
    )

    y_star = x_star_mv(θ)

    # Option 1
    I = jnp.eye(n_modes)

    """
    times = [time()]
    H = hess(θ, flux, ivar, lambda c: matvec(c, x, 1, n_modes))
    times.append(time())
    ATCinvA = rmatmat(ivar * matmat(I, x, 1, n_modes), x, 1, n_modes)
    times.append(time())
    print(jnp.max(jnp.abs(ATCinvA - H)))
    times.append(time())
    ATCinvA_inv, resid, rank, s = jnp.linalg.lstsq(H, I, rcond=None)
    times.append(time())
    """
    ATCinvA = rmatmat(ivar * matmat(I, x, C_inv_sqrt, n_modes), x, C_inv_sqrt, n_modes)
    ATCinvA_inv, resid, rank, s = jnp.linalg.lstsq(ATCinvA, I, rcond=None)

    A_star_mm = jax.vmap(x_star_mv, in_axes=(0, ))
    C_inv_star = 1/jnp.diag(A_star_mm(A_star_mm(ATCinvA_inv).T))

    """
    # Option 2
    ATCinvA_inv = jaxopt.linear_solve.solve_normal_cg(
        lambda v: ATCinvAv(v, x, C_inv_sqrt, n_modes), 
        jnp.eye(n_modes)
    )
    A_star_mm = jax.vmap(x_star_mv, in_axes=(0, ))
    C_inv_star = 1/jnp.diag(A_star_mm(A_star_mm(ATCinvA_inv).T))
    """
    
    """
    # Option 3
    @jax.jit
    def _ATCinvAv(c):
        return rmatvec(matvec(c, x, C_inv_sqrt, n_modes), x, C_inv_sqrt, n_modes)

    def ATCinvAv(c):
        return jax.vmap(_ATCinvAv, in_axes=(0, ))(c)

    # Optrion 3
    # Build the normal matrix column by column
    I = jnp.eye(n_modes)
    ATCinvA = ATCinvAv(I)
    ATCinvA_inv, *extras = jnp.linalg.lstsq(ATCinvA, I, rcond=rcond)
    """

    """
    # Option 4
    @jax.jit
    def _ATCinvAv(c):
        return rmatvec(matvec(c, x, C_inv_sqrt, n_modes), x, C_inv_sqrt, n_modes)

    
    # Replace the current ATCinvA computation with:
    def solve_ATCinvA_system(b):
        return jaxopt.linear_solve.solve_normal_cg(
            _ATCinvAv, 
            b, 
            init=jnp.zeros_like(b),
            maxiter=n_modes,
            tol=min(rtol, atol)
        )

    # Solve for each column of the identity matrix
    I = jnp.eye(n_modes)
    ATCinvA_inv = jax.vmap(solve_ATCinvA_system, in_axes=1, out_axes=1)(I)
    """

    """
    # Option 5
    # Use iterative solver instead of building full matrix
    def solve_for_covariance_diagonal():
        # Only compute diagonal elements of (A^T C^{-1} A)^{-1} A_star^T A_star
        A_star_columns = jax.vmap(lambda i: matvec(jnp.eye(n_modes)[i], x_star, 1, n_modes))(jnp.arange(n_modes))
        mv = lambda c: _ATCinvAv_efficient(c, x, C_inv_sqrt, n_modes)
        def solve_single_system(a_star_col):
            return jaxopt.linear_solve.solve_normal_cg(
                mv,
                a_star_col,
                init=jnp.zeros(n_modes),
                maxiter=min(n_modes, 1000),
                tol=rtol
            )
        
        solved_systems = jax.vmap(solve_single_system)(A_star_columns)
        # Diagonal of A_star (A^T C^{-1} A)^{-1} A_star^T
        diagonal_elements = jnp.sum(A_star_columns * solved_systems, axis=1)
        return 1.0 / diagonal_elements

    C_inv_star = solve_for_covariance_diagonal()
    """
    """
    # Option 6:
    diagonal_approx = ATCinvA_diag(x, C_inv_sqrt, n_modes)
    ATCinvA_inv_approx = ATCinvA_inv = jnp.diag(1.0 / diagonal_approx)
    """

    #A_star_mm = jax.vmap(x_star_mv, in_axes=(0, ))
    #C_inv_star = 1/jnp.diag(A_star_mm(A_star_mm(ATCinvA_inv).T))

    # Option 7
    """
    #C_inv_star = 1/compute_diagonal_efficient(x, x_star, n_modes, ivar)

    #def compute_diagonal_efficient(x, x_star, n_modes, C_inv_diag):

    def B_apply(c):
        return rmatvec(ivar * matvec(c, x, 1, n_modes), x, 1, n_modes)
    
    def compute_single_diagonal(e_i):
        
        # This gives us A_star[i, :] (i-th row)
        a_star_i_row = rmatvec(e_i, x_star, 1, n_modes)  # This is correct!
        
        # Solve B @ y = a_star_i_row  
        y_i = jax.scipy.sparse.linalg.cg(B_apply, a_star_i_row, maxiter=100)[0]
        
        # Diagonal element = a_star_i_row @ y_i
        return jnp.dot(a_star_i_row, y_i)
    
    diagonal = jax.vmap(compute_single_diagonal)(jnp.eye(n_modes))
    C_inv_star = 1.0 / diagonal
    """

    """
    # Option 8

    A_star_apply = jax.jit(partial(matvec, x=x_star, w=1, n_modes=n_modes))
    AT_apply = jax.jit(partial(rmatvec, x=x, w=1, n_modes=n_modes))
    A_apply = jax.jit(partial(matvec, x=x, w=1, n_modes=n_modes))
    A_starT_apply = jax.jit(partial(rmatvec, x=x_star, w=1, n_modes=n_modes))
    C_inv_diag = ivar

    def single_estimate(key):
        # Random vector (Rademacher or Gaussian)
        z = jax.random.rademacher(key, (n_modes,))
        
        # Compute (full_matrix @ z)
        def full_apply(x):
            temp1 = A_star_apply(x)
            temp2 = AT_apply(C_inv_diag * A_apply(temp1))
            y = jax.scipy.sparse.linalg.cg(lambda v: AT_apply(C_inv_diag * A_apply(v)), temp2)[0]
            return A_starT_apply(y)
        
        result = full_apply(z)
        return z * result  # Element-wise product
    
    key = jax.random.PRNGKey(0)

    keys = jax.random.split(key, 100)
    estimates = jax.vmap(single_estimate)(keys)

    C_inv_star = 2 * jnp.mean(estimates, axis=0)
    """

    return (y_star, C_inv_star, {})
    

def _pre_matvec(c, p):
    """
    Enforce Hermitian symmetry on the Fourier coefficients.

    :param c:
        The Fourier coefficients (real-valued).
    
    :param p:
        The number of modes.
    """
    f = (
        0.5  * jnp.hstack([c[:p//2+1],   jnp.zeros(p-p//2-1)])
    +   0.5j * jnp.hstack([jnp.zeros(p//2+1), c[p//2+1:]])
    )
    return f + jnp.conj(jnp.flip(f))

def _post_rmatvec(f, p):
    f_flat = f.flatten()
    return jnp.hstack([jnp.real(f_flat[:p//2+1]), jnp.imag(f_flat[-(p-p//2-1):])])

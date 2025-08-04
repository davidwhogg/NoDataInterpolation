
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from functools import partial
from sklearn.neighbors import KDTree
from typing import Optional

from jax_finufft import nufft2
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
        n_modes or len(λ_out), 
        rtol=rtol,
        atol=atol,
        rcond=rcond,
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


@partial(jax.jit, static_argnames=("n_modes", "rtol", "atol", "rcond"))
def _frizzle(λ_out, λ, flux, ivar, n_modes, rtol, atol, rcond):

    λ_all = jnp.hstack([λ, λ_out])
    λ_min, λ_max = (jnp.min(λ_all), jnp.max(λ_all))

    small = (λ_max - λ_min)/(1 + len(λ_out))
    scale = (1 - small) * 2 * jnp.pi / (λ_max - λ_min)
    x = (λ - λ_min) * scale
    x_star = (λ_out - λ_min) * scale

    C_inv_sqrt = jnp.sqrt(ivar)
    
    A_op = lx.FunctionLinearOperator(
        lambda c: jnp.real(nufft2(_pre_matvec(c, n_modes), x)),
        jax.ShapeDtypeStruct((n_modes, ), x.dtype)
    )
    C_inv_op = lx.DiagonalLinearOperator(C_inv_sqrt)

    A_w = C_inv_op @ A_op
    Y_w = C_inv_sqrt * flux

    solver = lx.NormalCG(rtol=rtol, atol=atol)
    solution = lx.linear_solve(A_w, Y_w, solver)

    θ = solution.value
    meta = dict(result=solution.result, **solution.stats)

    A_star_op = lx.FunctionLinearOperator(
        lambda c: jnp.real(nufft2(_pre_matvec(c, n_modes), x_star)),
        jax.ShapeDtypeStruct((n_modes, ), x.dtype)
    )

    y_star = A_star_op.mv(θ)

    nll = lambda θ: 0.5 * jnp.sum(ivar * (A_op.mv(θ) - flux)**2)
    hess = jax.hessian(nll)(θ)
    I = jnp.eye(n_modes)

    ATCinvA_inv, resid, rank, s = jnp.linalg.lstsq(hess, I, rcond=rcond)

    A_star_mm = jax.vmap(A_star_op.mv, in_axes=(0, ))
    C_inv_star = 1/jnp.diag(A_star_mm(A_star_mm(ATCinvA_inv).T))

    return (y_star, C_inv_star, meta)


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
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from sklearn.neighbors import KDTree
from typing import Optional

from jax_finufft import nufft1, nufft2
from .utils import check_inputs, combine_flags

def frizzle(
    λ_out: jnp.array,
    λ: jnp.array,
    flux: jnp.array,
    ivar: Optional[jnp.array] = None,
    mask: Optional[jnp.array] = None,
    flags: Optional[jnp.array] = None,
    censor_missing_regions: Optional[bool] = True,
    n_modes: Optional[int] = None,
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
            
    :returns:
        A four-length tuple of ``(flux, ivar, flags, meta)`` where:
            - ``flux`` is the combined fluxes,
            - ``ivar`` is the variance of the combined fluxes,
            - ``flags`` are the combined flags, and 
            - ``meta`` is a dictionary.
    """

    λ_out, λ, flux, ivar, mask = check_inputs(λ_out, λ, flux, ivar, mask)

    y_star, C_inv_star, meta = _frizzle_materialized(
        λ_out, λ[~mask], flux[~mask], ivar[~mask], 
        n_modes or min([len(λ_out), int(np.sum(~mask))]),
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


@partial(jax.jit, static_argnames=("n_modes",))
def matvec(c, x, n_modes):
    return jnp.real(nufft2(_pre_matvec(c, n_modes), x))

@partial(jax.jit, static_argnames=("n_modes",))
def rmatvec(f, x, n_modes):
    dtype = jnp.array(0.0 + 0.0j).dtype
    return _post_rmatvec(nufft1(n_modes, f.astype(dtype), x), n_modes)

@partial(jax.jit, static_argnames=("n_modes",))
def ATCinvA(c, x, n_modes):
    return rmatvec(matvec(c, x, n_modes), x, n_modes)

matmat = jax.vmap(matvec, in_axes=(0, None, None))
rmatmat = jax.vmap(rmatvec, in_axes=(0, None, None))

@partial(jax.jit, static_argnames=("n_modes", ))
def _frizzle_materialized(λ_out, λ, flux, ivar, n_modes):
    """
    frizzle some input spectra using materialized matrices.
    """

    λ_min, λ_max = (λ_out[0], λ_out[-1])

    small = (λ_max - λ_min)/(1 + len(λ_out))
    scale = (1 - small) * 2 * jnp.pi / (λ_max - λ_min)
    x = (λ - λ_min) * scale 
    x_star = (λ_out - λ_min) * scale

    I = jnp.eye(n_modes)
    ATCinv = matmat(I, x, n_modes) * ivar
    ATCinvA = rmatmat(ATCinv, x, n_modes)

    cho_factor = jax.scipy.linalg.cho_factor(ATCinvA)
    θ = jax.scipy.linalg.cho_solve(cho_factor, ATCinv @ flux, check_finite=False)
    ATCinvA_inv = jax.scipy.linalg.cho_solve(cho_factor, I, check_finite=False)

    y_star = matvec(θ, x_star, n_modes)
    A_star = matmat(I, x_star, n_modes)
    C_inv_star = 1/jnp.diag(A_star @ ATCinvA_inv @ A_star.T)
    return (y_star, C_inv_star, {})
    

@partial(jax.jit, static_argnames=("p", ))
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

@partial(jax.jit, static_argnames=("p",))
def _post_rmatvec(f, p):
    f_flat = f.flatten()
    return jnp.hstack([jnp.real(f_flat[:p//2+1]), jnp.imag(f_flat[-(p-p//2-1):])])


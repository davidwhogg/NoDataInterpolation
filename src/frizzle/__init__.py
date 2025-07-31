
import numpy as np
import numpy.typing as npt
import warnings
from pylops import Diagonal
from scipy.sparse.linalg import lsqr
from sklearn.neighbors import KDTree
from typing import Optional
from jax import (numpy as jnp, hessian)
from nifty_solve.jax_operators import JaxFinufft1DRealOperator

from .utils import ensure_dict, check_inputs, combine_flags

def frizzle(
    λ_out: npt.ArrayLike,
    λ: npt.ArrayLike,
    flux: npt.ArrayLike,
    ivar: Optional[npt.ArrayLike] = None,
    mask: Optional[npt.ArrayLike] = None,
    flags: Optional[npt.ArrayLike] = None,
    censor_missing_regions: Optional[bool] = True,
    n_modes: Optional[int] = None,
    lsqr_kwds: Optional[dict] = None,
    finufft_kwds: Optional[float] = None,
    **kwargs
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
            
    :param finufft_kwds: [optional]
        Keyword arguments to pass to the `finufft.Plan()` constructor.
    
    :param lsqr_kwds: [optional]
        Keyword arguments to pass to the `scipy.sparse.linalg.lsqr()` function.
    
    :returns:
        A four-length tuple of ``(flux, ivar, flags, meta)`` where:
            - ``flux`` is the combined fluxes,
            - ``ivar`` is the variance of the combined fluxes,
            - ``flags`` are the combined flags, and 
            - ``meta`` is a dictionary.
    """

    n_modes = n_modes or len(λ_out)
    lsqr_kwds = ensure_dict(lsqr_kwds, calc_var=False)
    finufft_kwds = ensure_dict(finufft_kwds)

    λ_out, λ, flux, ivar, mask = check_inputs(λ_out, λ, flux, ivar, mask)

    λ_all = jnp.hstack([λ[~mask], λ_out])
    λ_min, λ_max = (jnp.min(λ_all), jnp.max(λ_all))

    small = kwargs.get("small", 1e-5)
    scale = (1 - small) * 2 * jnp.pi / (λ_max - λ_min)
    x = (λ[~mask] - λ_min) * scale
    x_star = (λ_out - λ_min) * scale

    C_inv_sqrt = jnp.sqrt(ivar[~mask])

    A = JaxFinufft1DRealOperator(x, n_modes, **finufft_kwds)

    A_w = Diagonal(C_inv_sqrt) @ A
    Y_w = C_inv_sqrt * flux[~mask]
    θ, *extras = lsqr(A_w, Y_w, **lsqr_kwds)
    
    keys = (
        "istop", "itn", "r1norm", "r2norm", "anorm", "acond", "arnorm", 
        "xnorm", "var"
    )
    meta = dict(zip(keys, extras))

    A_star = JaxFinufft1DRealOperator(x_star, n_modes, **finufft_kwds)
    y_star = np.array(A_star @ θ)
    
    nll = lambda θ: 0.5 * jnp.sum(ivar[~mask] * (A @ θ - flux[~mask])**2)
    
    hess = hessian(nll)(θ)
    I = jnp.eye(n_modes)

    # When resampling a single epoch, you might have a bad time.
    for rcond in [None, 1e-15, 1e-12, 1e-9, 1e-6]:
        C, *extras = jnp.linalg.lstsq(hess, I, rcond=rcond)
        if jnp.min(jnp.diag(C)) > 0:
            break

    if rcond is not None:
        warnings.warn(f"Condition number of C is high, rcond={rcond:.2e}")

    meta["rcond"] = rcond

    C_inv_star = 1/np.diag(A_star @ (A_star @ C).T)
    
    if censor_missing_regions:
        # Set NaNs for regions where there were NO data.
        # Here we check to see if the closest input value was more than the output pixel width.
        tree = KDTree(x.reshape((-1, 1)))
        distances, indices = tree.query(x_star.reshape((-1, 1)), k=1)

        no_data = np.hstack([distances[:-1, 0] > np.diff(x_star), False])
        meta["no_data_mask"] = no_data
        if np.any(no_data):
            y_star[no_data] = np.nan
            C_inv_star[no_data] = 0
    
    flags_star = combine_flags(λ_out, λ, flags)

    return (y_star, C_inv_star, flags_star, meta)
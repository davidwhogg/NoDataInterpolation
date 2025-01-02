import finufft
import numpy as np
import numpy.typing as npt
import warnings
from pylops import LinearOperator, MatrixMult, Diagonal, Identity
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsqr
from sklearn.neighbors import KDTree
from typing import Optional, Union, Tuple

from .operator import FrizzleOperator, Diagonal
from .utils import ensure_dict, check_inputs, separate_flags, combine_flags

def frizzle(
    λ_out: npt.ArrayLike,
    λ: npt.ArrayLike,
    flux: npt.ArrayLike,
    ivar: Optional[npt.ArrayLike] = None,
    mask: Optional[npt.ArrayLike] = None,
    flags: Optional[npt.ArrayLike] = None,
    n_modes: Optional[int] = None,
    n_uncertainty_samples: Optional[Union[int, float]] = None,
    censor_missing_regions: Optional[bool] = True,
    lsqr_kwds: Optional[dict] = None,
    finufft_kwds: Optional[float] = None,
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
    
    :param n_modes: [optional]
        The number of Fourier modes to use. If `None` is given then this will default to `len(λ_out)`.
    
    :param n_uncertainty_samples: [optional]
        The number of samples to use when estimating the uncertainty of the combined spectrum by Hutchinson's method. 
        If `None` is given then this will default to `0.10 * n_modes`. If a float is given then this will be interpreted as a fraction 
        of `n_modes`. If a number is given that exceeds `n_modes`, the uncertainty will be computed exactly (slow).

    :param censor_missing_regions: [optional]
        If `True`, then regions where there is no data will be set to NaN in the combined spectrum. If `False` the values evaluated
        from the model will be reported (and have correspondingly large uncertainties) but this will produce unphysical features.
        
    :param finufft_kwds: [optional]
        Keyword arguments to pass to the `finufft.Plan()` constructor.
    
    :param lsqr_kwds: [optional]
        Keyword arguments to pass to the `scipy.sparse.linalg.lsqr()` function.

        The most relevant `lsqr()` keyword to the user is `calc_var`, which will compute the variance of the combined spectrum.
        By default this is set to `True`, but it can be set to `False` to speed up the computation if the variance is not needed.
    
    :returns:
        A four-length tuple of ``(flux, ivar, flags, meta)`` where:
            - ``flux`` is the combined fluxes,
            - ``ivar`` is the variance of the combined fluxes,
            - ``flags`` are the combined flags, and 
            - ``meta`` is a dictionary.
    """

    n_modes = n_modes or len(λ_out)
    lsqr_kwds = ensure_dict(lsqr_kwds)
    finufft_kwds = ensure_dict(finufft_kwds)

    λ_out, λ, flux, ivar, mask = check_inputs(λ_out, λ, flux, ivar, mask)

    λ_all = np.hstack([λ[~mask], λ_out])
    λ_min, λ_max = (np.min(λ_all), np.max(λ_all))

    # This is setting the scale to be such that the Fourier modes are in the range [0, 2π).
    small = 1e-5
    scale = (1 - small) * 2 * np.pi / (λ_max - λ_min)
    x = (λ[~mask] - λ_min) * scale
    x_star = (λ_out - λ_min) * scale
    
    C_inv_sqrt = np.sqrt(ivar[~mask])

    A = FrizzleOperator(x, n_modes, **finufft_kwds)
    C_inv = Diagonal(np.ascontiguousarray(ivar[~mask]))

    A_w = Diagonal(C_inv_sqrt) @ A
    Y_w = C_inv_sqrt * flux[~mask]
    θ, *extras = lsqr(A_w, Y_w, **lsqr_kwds)
    
    meta = dict(zip(["istop", "itn", "r1norm", "r2norm", "anorm", "acond", "arnorm", "xnorm", "var"], extras))
    

    A_star = FrizzleOperator(x_star, n_modes, **finufft_kwds)
    y_star = A_star @ θ

    # Until I know what lsqr `var` is doing..
    ATCinvA_inv = lsqr(A.T @ C_inv @ A, np.ones(n_modes), **lsqr_kwds)[0]
    Op = (A_star @ Diagonal(ATCinvA_inv) @ A_star.T)

    n_uncertainty_samples = n_uncertainty_samples or 0.1
    if n_uncertainty_samples < 1:
        n_uncertainty_samples *= n_modes
        
    n_uncertainty_samples = int(n_uncertainty_samples)        
    if n_uncertainty_samples < n_modes:
        # Estimate the diagonals with Hutchinson's method.
        v = np.random.randn(n_modes, n_uncertainty_samples)
        C_inv_star = n_uncertainty_samples/np.sum((Op @ v) * v, axis=1)
    else:            
        C_inv_star = 1/np.diag(Op.todense())
    
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
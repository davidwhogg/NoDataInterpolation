import finufft
import numpy as np
import numpy.typing as npt
import warnings
from pylops import LinearOperator, MatrixMult, Diagonal, Identity
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsqr, lsmr, gmres
from sklearn.neighbors import KDTree
from typing import Optional, Union, Tuple


from .operator import FrizzleOperator, Diagonal
from .utils import ensure_dict, check_inputs, separate_flags


def frizzle(
    λ_out: npt.ArrayLike,
    λ: npt.ArrayLike,
    flux: npt.ArrayLike,
    ivar: Optional[npt.ArrayLike] = None,
    mask: Optional[npt.ArrayLike] = None,
    flags: Optional[npt.ArrayLike] = None,
    n_modes: Optional[int] = None,
    censor_missing_regions: Optional[bool] = True,
    lsqr_kwds: Optional[dict] = None,
    finufft_kwds: Optional[float] = None,
):
    """
    Combine spectra by forward modelling.

    :param λ_out:
        The wavelengths to sample the combined spectrum on. This should be shape (M_star, ).
    
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
        The number of Fourier modes to use. If `None` is given then this will default to `M_star`.
    
    :param censor_missing_regions: [optional]
        If `True`, then regions where there is no data will be set to NaN in the resampled spectrum. If `False` the values evaluated
        from the model will be reported (and have correspondingly large uncertainties) but this will produce unphysical features.

    :param finufft_kwds: [optional]
        Keyword arguments to pass to the `finufft.Plan()` constructor.
    
    :param lsqr_kwds: [optional]
        Keyword arguments to pass to the `scipy.sparse.linalg.lsqr()` function.
    
    :returns:
        A four-length tuple of ``(flux, ivar, flags, meta)`` where ``flux`` is the resampled flux values 
    and ``ivar`` is the variance of the resampled fluxes, and ``flags`` are the resampled flags, and ``meta`` is a dictionary.
    """

    n_modes = n_modes or len(λ_out)
    lsqr_kwds = ensure_dict(lsqr_kwds, calc_var=True)
    finufft_kwds = ensure_dict(finufft_kwds)

    λ, flux, ivar, mask = check_inputs(λ_out, λ, flux, ivar, mask)

    λ_all = np.hstack([λ[~mask], λ_out])
    λ_min, λ_max = (np.min(λ_all), np.max(λ_all))

    # This is setting the scale to be such that the Fourier modes are in the range [0, 2π].
    scale = 2 * np.pi / (λ_max - λ_min)
    x = (λ[~mask] - λ_min) * scale
    x_star = (λ_out - λ_min) * scale

    A = FrizzleOperator(x, n_modes, **finufft_kwds)
    Y = flux[~mask]
    C_inv = Diagonal(np.ascontiguousarray(ivar[~mask]))
    ATCinv = A.T @ C_inv

    θ, *extras = lsqr(ATCinv @ A, ATCinv @ Y, **lsqr_kwds)
    meta = dict(zip(["istop", "itn", "r1norm", "r2norm", "anorm", "acond", "arnorm", "xnorm", "var"], extras))
    
    A_star = FrizzleOperator(x_star, n_modes, **finufft_kwds)
    y_star = A_star @ θ

    # This is the cheapest way to compute the inverse variance of the resampled spectrum.
    if lsqr_kwds.get("calc_var", False):
        C_inv_star = 1/np.diag((A_star @ Diagonal(meta["var"]**0.5) @ A_star.T).todense())
    else:
        C_inv_star = np.zeros_like(y_star)

    # Alternatively:
    #ATCinvA_inv, *_ = lstsq((A.T @ C_inv @ A).todense(), np.eye(n_modes))
    #C_star = np.diag((A_star @ MatrixMult(ATCinvA_inv) @ A_star.T).todense())
    
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


def combine_flags(x_star, x, flags):
    flags_star = np.zeros(x_star.size, dtype=np.uint64 if flags is None else flags.dtype)
    x_star_T = x_star.reshape((-1, 1))
    diff_x_star = np.diff(x_star)
    for bit, flag in separate_flags(flags).items():
        tree = KDTree(x[flag].reshape((-1, 1)))            
        distances, indices = tree.query(x_star_T, k=1)
        within_pixel = np.hstack([distances[:-1, 0] <= diff_x_star, False])
        flags_star[within_pixel] += 2**bit
    return flags_star

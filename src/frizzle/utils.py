import numpy as np
import numpy.typing as npt
from typing import Optional
from sklearn.neighbors import KDTree

def ensure_dict(d, **defaults):
    kwds = dict()
    kwds.update(defaults)
    kwds.update(d or {})
    return kwds

def check_inputs(λ_out, λ, flux, ivar, mask):
    λ, flux = map(np.hstack, (λ, flux))
    if mask is None:
        mask = np.zeros(flux.size, dtype=bool)
    else:
        mask = np.hstack(mask).astype(bool)
    
    λ_out = np.array(λ_out)
    # Mask things outside of the resampling range
    mask *= ((λ_out[0] <= λ) * (λ <= λ_out[-1]))

    if ivar is None:
        ivar = np.ones_like(flux)
    else:
        ivar = np.hstack(ivar)    
    return (λ_out, λ, flux, ivar, mask)

def separate_flags(flags: Optional[npt.ArrayLike] = None):
    """
    Separate flags into a dictionary of flags for each bit.
    
    :param flags:
        An ``M``-length array of flag values.
    
    :returns:
        A dictionary of flags, where each key is a bit and each value is an array of 0s and 1s.
    """
    separated = {}
    if flags is not None:
        for q in range(1 + int(np.log2(np.max(flags)))):
            is_set = (flags & np.uint64(2**q)) > 0
            if any(is_set):
                separated[q] = is_set.astype(bool)
    return separated    

def combine_flags(λ_out, λ, flags):
    """
    Combine flags from input spectra.

    :param λ_out:
        The wavelengths to sample the combined spectrum on.
    
    :param λ:
        The input wavelengths.
    
    :param flags:
        An array of integer flags.
    """
    flags_star = np.zeros(λ_out.size, dtype=np.uint64 if flags is None else flags.dtype)
    λ_out_T = λ_out.reshape((-1, 1))
    diff_λ_out = np.diff(λ_out)
    for bit, flag in separate_flags(flags).items():
        tree = KDTree(λ[flag].reshape((-1, 1)))            
        distances, indices = tree.query(λ_out_T, k=1)
        within_pixel = np.hstack([distances[:-1, 0] <= diff_λ_out, False])
        flags_star[within_pixel] += 2**bit
    return flags_star
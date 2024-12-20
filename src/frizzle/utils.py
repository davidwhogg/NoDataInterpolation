import numpy as np
import numpy.typing as npt
from typing import Optional


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
    
    # Mask things outside of the resampling range
    mask *= ((λ_out[0] <= λ) * (λ <= λ_out[-1]))

    if ivar is None:
        ivar = np.ones_like(flux)
    else:
        ivar = np.hstack(ivar)    
    return (λ, flux, ivar, mask)


    
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


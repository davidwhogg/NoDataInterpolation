import numpy as np
import warnings
from collections import OrderedDict
from typing import List, Tuple, Optional
from itertools import cycle

def resample_spectrum(
    resample_wavelength: np.array,
    wavelength: np.array,
    flux: np.array,
    ivar: Optional[np.array] = None,
    flags: Optional[np.array] = None,
    mask: Optional[np.array] = None,
    L: Optional[int] = None,
    P: Optional[int] = None,
    X_star: Optional[np.array] = None,
    min_resampled_flag_value: Optional[float] = 0.1,
    Lambda: Optional[float] = 0.0,
    rcond: Optional[float] = 1e-15,
    full_output: Optional[bool] = False,
) -> Tuple[np.array, np.array]:
    """
    Sample a spectrum on a wavelength array given a set of pixels recorded from one or many visits.
    
    :param resample_wavelength:
        A ``M_star``-length array of wavelengths to sample the spectrum on. In the paper, this is equivalent 
        to the output-spectrum pixel grid $x_\star$.

    :param wavelength:
        A ``M``-length array of wavelengths from individual visits. If you have $N$ spectra where 
        the $i$th spectrum has $m_i$ pixels, then $M = \sum_{i=1}^N m_i$, and this array represents a
        flattened 1D array of all wavelength positions. In the paper, this is equivalent to the 
        input-spectrum pixel grid $x_i$.
    
    :param flux:
        A ``M``-length array of flux values from individual visits. In the paper, this is equivalent to 
        the observations $y_i$.
    
    :param ivar: [optional]
        A ``M``-length array of inverse variance values from individual visits. In the paper, this is 
        equivalent to the individual inverse variance matrices $C_i^{-1}$.

    :param flags: [optional]
        A ``M``-length array of bitmask flags from individual visits.
    
    :param mask: [optional]
        A ``M``-length array of boolean values indicating whether a pixel should be used or not in
        the resampling (`True` means mask the pixel, `False` means use the pixel). If `None` is
        given then all pixels will be used. The `mask` is only relevant for sampling the flux and
        inverse variance values, and not the flags.
    
    :param X_star: [optional]
        The design matrix to use when solving for the combined spectrum. If you are resampling
        many spectra to the same wavelength array then you will see performance improvements by
        pre-computing this design matrix and supplying it. To pre-compute it:

        ```python
        X_star = construct_design_matrix(resample_wavelength, L, P)
        ```
    
        Then supply `X_star` to this function, and optionally `L` and `P` to ensure consistency.
    
    :param L: [optional]
        The length scale for the Fourier modes. 
        If `None` is given, this will default to the peak-to-peak range of `wavelength`.

    :param P: [optional]
        The number of Fourier modes to use when solving for the resampled spectrum.
        If `None` is given, this will default to the number of pixels in `wavelength`.    
    
    :param min_resampled_flag_value: [optional]
        The minimum value of a flag to be considered "set" in the resampled spectrum. This is
        used to reconstruct the flags in the resampled spectrum. The default is 0.1, but a
        sensible choice could be 1/N, where N is the number of visits.    
    
    :param Lambda: [optional]
        An optional regularization strength.
    
    :param rcond: [optional]
        Cutoff for small singular values. Singular values less than or equal to 
        ``rcond * largest_singular_value`` are set to zero (default: 1e-15).
                
    :param full_output: [optional]
        If `True`, a number of additional outputs will be returned. These are:
        - `sampled_separate_flags`: A dictionary of flags, where each key is a bit and each value
            is an array of 0s and 1s.
        - `X_star`: The design matrix used to solve for the resampled spectrum.
        - `L`: The length scale used to solve for the resampled spectrum.
        - `P`: The number of Fourier modes used to solve for the resampled spectrum.
        
    :returns:
        A three-length tuple of ``(flux, ivar, flags)`` where ``flux`` is the resampled flux values 
        and ``ivar`` is the variance of the resampled fluxes, and ``flags`` are the resampled flags.
        
        All three arrays are length $M_\star$ (the same as ``resample_wavelength``). If ``full_output``
        is `True`, then the tuple will be length 7 with the additional outputs specified above.
    """

    linalg_kwds = dict(Lambda=Lambda, rcond=rcond)
    wavelength, flux, ivar, mask = _check_shapes(wavelength, flux, ivar, mask)

    resample_wavelength = np.array(resample_wavelength)

    # Restrict the resampled wavelength to the range of the visit spectra.
    sampled_wavelengths = wavelength[(ivar > 0) & (~mask)]
    min_sampled_wavelength, max_sampled_wavelength = (np.min(sampled_wavelengths), np.max(sampled_wavelengths))

    is_sampled = (max_sampled_wavelength >= resample_wavelength) * (resample_wavelength >= min_sampled_wavelength)
    x_star = resample_wavelength[is_sampled]

    min_wavelength, max_wavelength = x_star[[0, -1]]
    L = L or (max_wavelength - min_wavelength)
    P = P or len(x_star)

    # We need to construct the design matrices to only be restricted by wavelength.
    # Then for flux values we will use the `mask` to restrict the flux values.
    # For flags, we will not use any masking.
    use_pixels = (
        (max_wavelength >= wavelength) 
    *   (wavelength >= min_wavelength) 
    &   (~mask)
    )

    X = construct_design_matrix(wavelength, L, P) # M x P
    Y = flux
    Cinv = ivar

    XTCinvX_inv, XTC_invX_invXTCinv = _XTCinvX_invXTCinv(
        X[use_pixels], 
        Cinv[use_pixels], 
        **linalg_kwds
    )

    if X_star is None:
        X_star = construct_design_matrix(x_star, L, P)

    A_star_masked = X_star @ XTC_invX_invXTCinv
    y_star_masked = A_star_masked @ Y[use_pixels]
    Cinv_star_masked = 1/np.diag(X_star @ XTCinvX_inv @ X_star.T)
    if np.any(Cinv_star_masked < 0):
        warnings.warn(
            "Clipping negative inverse variances to zero. It is likely that the "
            "requested wavelength range to resample is wider than the visit spectra."
        )
        Cinv_star_masked = np.clip(Cinv_star_masked, 0, None)

    separate_flags = OrderedDict()
    flags_star_masked = np.zeros(x_star.size, dtype=np.uint64)

    if flags is not None:
        _, XTC_invX_invXTCinv = _XTCinvX_invXTCinv(X, Cinv, **linalg_kwds)
        A_star = X_star @ XTC_invX_invXTCinv

        separated_flags = _separate_flags(flags)
        for bit, flag in separated_flags.items():
            separate_flags[bit] = A_star @ flag
                    
        # Reconstruct flags
        for k, values in separate_flags.items():
            flag = (values > min_resampled_flag_value).astype(int)
            flags_star_masked += (flag * (2**k)).astype(flags_star_masked.dtype)

    # De-mask.
    # TODO: Should we return 0 fluxes as default, or NaNs? I think NaNs is better and 0 ivar.
    y_star = _un_mask(y_star_masked, is_sampled, default=np.nan)
    ivar_star = _un_mask(Cinv_star_masked, is_sampled, default=0)
    flags_star = _un_mask(flags_star_masked, is_sampled, default=0, dtype=np.uint64)

    if full_output:
        return (y_star, ivar_star, flags_star, separate_flags, X_star, L, P)
    else:
        return (y_star, ivar_star, flags_star)
    

def _un_mask(values, mask, default, dtype=float):
    v = default * np.ones(mask.shape, dtype=dtype)
    v[mask] = values
    return v


def _separate_flags(flags: np.array):
    """
    Separate flags into a dictionary of flags for each bit.
    
    :param flags:
        An ``M``-length array of flag values.
    
    :returns:
        A dictionary of flags, where each key is a bit and each value is an array of 0s and 1s.
    """
    separated = OrderedDict()
    for q in range(1 + int(np.log2(np.max(flags)))):
        is_set = (flags & np.uint64(2**q)) > 0
        separated[q] = np.clip(is_set, 0, 1)
    return separated    


def construct_design_matrix(wavelength: np.array, L: float, P: int):
    """
    Take in a set of wavelengths and return the Fourier design matrix.

    :param wavelength:
        An ``M``-length array of wavelength values.
        
    :param L:
        The length scale, usually taken as ``max(wavelength) - min(wavelength)``.

    :param P:
        The number of Fourier modes to use.
    
    :returns:
        A design matrix of shape (M, P).
    """
    # TODO: This could be replaced with something that makes use of finufft.
    scale = (np.pi * wavelength) / L
    X = np.ones((wavelength.size, P), dtype=float)
    for j, f in zip(range(1, P), cycle((np.sin, np.cos))):
        X[:, j] = f(scale * (j + (j % 2)))
    return X


def _XTCinvX_invXTCinv(X, Cinv, Lambda=0, rcond=1e-15):
    N, P = X.shape
    XTCinv = X.T * Cinv
    XTCinvX = XTCinv @ X
    XTCinvX += Lambda
    XTCinvX_inv = np.linalg.pinv(XTCinvX, rcond=rcond)
    XTCinvX_invXTCinv = XTCinvX_inv @ XTCinv
    return (XTCinvX_inv, XTCinvX_invXTCinv)


def _check_shape(name, a, P):
    a = np.array(a)
    if a.size != P:
        raise ValueError(f"{name} must be the same size as wavelength")
    return a


def _check_shapes(wavelength, flux, ivar, mask):
    wavelength = np.array(wavelength)

    P = wavelength.size
    flux = _check_shape("flux", flux, P)
    if ivar is not None:
        ivar = _check_shape("ivar", ivar, P)
    else:
        ivar = np.ones_like(flux)
    if mask is not None:
        mask = _check_shape("mask", mask, P).astype(bool)
    else:
        mask = np.zeros(flux.shape, dtype=bool)
    return (wavelength, flux, ivar, mask)
        


if __name__ == "__main__":

    # This is instantaneous now..
    x_min = 8.7000 # minimum ln wavelength
    x_max = 8.7025 # maximum ln wavelength

    # So let's test with 6750 pixels, which is something we could NOT afford when we had to construct all the 2D arrays slowly.
    #x_min = 8.7000 # minimum ln wavelength
    #x_max = 8.7500 # maximum ln wavelength

    import numpy as np
    import pylab as plt
    import scipy.interpolate as interp
    from time import time

    np.random.seed(34)
    c = 299792458. # m / s
    sqrt2pi = np.sqrt(2. * np.pi)


    R = 1.35e5 # resolution
    dxstar = 1. / R # output pixel grid spacing
    xstar = np.arange(x_min + 0.5 * dxstar, x_max, dxstar) # output pixel grid
    P = np.round((x_max - x_min) * R).astype(int) # number of Fourier modes (ish)


    # define high-level parameters, especially including spectrograph parameters
    sigma_x = 1. / R # LSF sigma in x units
    lines_per_x = 2.0e4 # mean density (Poisson rate) of lines per unit ln wavelength
    ew_max_x = 3.0e-5 # maximum equivalent width in x units
    ew_power = 5.0 # power parameter in EW maker
    badfrac = 0.01 # fraction of data to mark bad    

    # Set the pixel grid and model complexity for the output combined spectrum
    Mstar = len(xstar) # number of output pixels

    # set up the line list for the true spectral model
    x_margin = 1.e6/c # hoping no velocities are bigger than 1000 km/s
    x_range = x_max - x_min + 2. * x_margin # make lines in a bigger x range than the data range
    nlines = np.random.poisson(x_range * lines_per_x) # set the total number of lines
    line_xs = (x_min - x_margin) + x_range * np.random.uniform(size=nlines)

    # give those lines equivalent widths from a power-law distribution
    line_ews = ew_max_x * np.random.uniform(size=nlines) ** ew_power # don't ask


    # make the synthetic spectrum (spectral expectation), and also add noise

    def oned_gaussian(dxs, sigma):
        return np.exp(-0.5 * dxs ** 2 / sigma ** 2) / (sqrt2pi * sigma)

    def true_spectrum(xs, doppler, lxs=line_xs, ews=line_ews, sigma=sigma_x):
        """
        """
        return np.exp(-1. * np.sum(ews[None, :] *
                                oned_gaussian(xs[:, None] - doppler
                                                - lxs[None, :], sigma), axis=1))

    def ivar(ys, continuum_ivar):
        return continuum_ivar / ys

    def noisy_true_spectrum(xs, doppler, continuum_ivar):
        """
        """
        ys_true = true_spectrum(xs, doppler)
        y_ivars = ivar(ys_true, continuum_ivar)
        return  ys_true + np.random.normal(size=xs.shape) / np.sqrt(y_ivars), y_ivars

    def doppler_information(xs, doppler, continuum_ivar, dx):
        """
        # Bugs:
        - Horrifying numerical derivative!
        """
        dys_dx = (true_spectrum(xs, doppler + dx)
                - true_spectrum(xs, doppler - dx)) / (2. * dx)
        y_ivars = ivar(true_spectrum(xs, doppler), continuum_ivar)
        return np.sum(y_ivars * dys_dx ** 2)

    def badify(yy):
        """
        Make bad-pixel masks and badify the bad pixels.
        """
        bady = 1. * yy
        bs = (np.random.uniform(size=len(bady)) > badfrac).astype(int)
        bs = np.minimum(bs, np.roll(bs, 1))
        bs = np.minimum(bs, np.roll(bs, -1))
        nbad = np.sum(bs < 0.5)
        if nbad > 0:
            bady[bs < 0.5] += 2. * np.random.uniform(size=nbad)
        return bs, bady

    def make_one_dataset(dx, SNR, N=8):
        # create true Doppler shifts on a sinusoid of epoch number
        Delta_xs = (3.e4 / c) * np.cos(np.arange(N) / 3.)
        # set the ivar
        continuum_ivar = SNR ** 2 # inverse variance of the noise in the continuum
        # now make the noisy fake data
        xs = np.arange(x_min - 0.5 * dx, x_max + dx, dx)
        ys = np.zeros((N, len(xs)))
        y_ivars = np.zeros_like(ys)
        bs = np.zeros_like(ys).astype(int)
        for j in range(N):
            ys[j], y_ivars[j] = noisy_true_spectrum(xs, Delta_xs[j], continuum_ivar)
            bs[j], ys[j] = badify(ys[j])
        return xs, ys, y_ivars, bs, Delta_xs


    dx2 = 1. / R # pixel spacing in the poorly sampled data; UNDER-SAMPLED!
    SNR2 = 12. # s/n ratio per pixel in the continuum
    xs2, ys2, y_ivars2, bs2, Delta_xs2 = make_one_dataset(dx2, SNR2, N=2)
    M2 = len(xs2)
    N2 = len(ys2)
    name2 = "well sampled input"
    ys_true = true_spectrum(xs2, 0)

    wavelength = []
    flux = []
    ivar = []
    flags = []
    visit_spectra = []
    for i in range(ys2.shape[0]):
        visit_spectra.append(
            (
                xs2 - Delta_xs2[i],
                ys2[i],
                y_ivars2[i],
                1 - bs2[i], # In my world, 1 means bad and 0 means OK
            )
        )
        wavelength.extend(xs2 - Delta_xs2[i])
        flux.extend(ys2[i])
        ivar.extend(y_ivars2[i])
        flags.extend(1 - bs2[i])

    resample_wavelength = xstar

    bitmask_flags = np.random.choice(1025, size=len(flux))
    mask = (np.array(flags) > 0)


    resampled_flux, resampled_ivar, resampled_flags = resample_spectrum(resample_wavelength, wavelength, flux, ivar, bitmask_flags, mask=mask)


    fig, ax = plt.subplots()
    ax.plot(resample_wavelength, resampled_flux, c="k", drawstyle="steps-mid")

    ax.fill_between(
        resample_wavelength,
        resampled_flux - np.sqrt(1/resampled_ivar),
        resampled_flux + np.sqrt(1/resampled_ivar),
        facecolor="#666666",
        zorder=-1,
        step="mid"
    )
    ax.plot(xs2, ys_true, c="tab:blue")
    ax.plot(xstar, resampled_flux - true_spectrum(xstar, 0), c="tab:blue", drawstyle="steps-mid")


    for i in range(ys2.shape[0]):
        ax.plot(xs2 - Delta_xs2[i], ys2[i] + i + 1, c="k")
    fig.savefig("sampling.pdf")

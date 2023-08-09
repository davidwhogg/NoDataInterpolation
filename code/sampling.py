import numpy as np
from typing import List, Tuple, Optional
from itertools import cycle


def sample_combined_spectrum(
    wavelength: np.array,
    visit_spectra: List[Tuple[np.array, np.array, np.array, Optional[np.array]]],
    design_matrix: Optional[np.array] = None,
    P: Optional[int] = None,
    L: Optional[int] = None,
) -> Tuple[np.array, np.array]:
    """
    Sample a combined spectrum on a wavelength array given a series of visit spectra.

    The visit spectra are assumed to have no covariance between neighbouring pixels.

    :param wavelength:
        A ``N``-length array of wavelengths to sample the combined spectrum on.
    
    :param visit_spectra:
        A list of visit spectra, each of which is a tuple of ``(wavelength, flux, ivar, flags)``,
        where:
         - `wavelength` is an array of **rest-frame wavelengths**,
         - `flux` is an array of flux values,
         - `ivar` is an array of inverse variance of the flux values, and
         - `flags` is an optional array of bitmask flags, where 1 indicates a bad pixel.
    
    :param P: [optional]
        The number of Fourier modes to use when solving for the combined spectrum.
        If `None` is given, this will default to the number of pixels in `wavelength`.    
    
    :param L: [optional]
        The length scale for the Fourier modes. 
        If `None` is given, this will default to the peak-to-peak range of `wavelength`.
    
    :param design_matrix: [optional]
        The design matrix to use when solving for the combined spectrum.
        
        If you are sampling a large number of spectra on the same ``wavelength`` then this can be
        pre-computed:

        ```python
        design_matrix = construct_design_matrix(wavelength, P, L)
        ```

        If you supply a pre-computed ``design_matrix``, make sure you also give the ``wavelength``
        array used to compute it (and optionally, the correct ``P`` and ``L``).

    :returns:
        A two-length tuple of ``(flux, variance)`` where ``flux`` is the flux values of the combined
        spectrum, and ``variance`` is the variance of the combined spectrum. 
        
        Both ``flux`` and ``variance`` are length ``N`` (the same as ``wavelength``).
    """

    # TODO: Does not yet return an array of resampled flags.
    # TODO: Implicitly assumes that the wavelength array is sorted in ascending order.

    P = P or wavelength.size
    min_wavelength, max_wavelength = wavelength[[0, -1]]
    L = L or (max_wavelength - min_wavelength)
    if design_matrix is None:
        design_matrix = construct_design_matrix(wavelength, P, L)

    A, Y, CI = pack_all_matrices(
        visit_spectra,
        min_wavelength=min_wavelength,
        max_wavelength=max_wavelength,
        P=P,
        L=L
    )

    ATCinv = A.T * CI
    ATCinvA = ATCinv @ A
    ATCinvAinv = np.linalg.solve(ATCinvA, np.eye(P))
    X = ATCinvAinv @ ATCinv @ Y
    y = design_matrix @ X
    y_var = np.diag(design_matrix @ ATCinvAinv @ design_matrix.T)
    return (y, y_var)


def pack_all_matrices(visit_spectra, min_wavelength, max_wavelength, P, L):
    """
    Pack all matrices for a set of visit spectra.
    
    :param visit_spectra:
        A list of visit spectra, each of which is a tuple of ``(wavelength, flux, ivar, flags)``,
        where:
         - `wavelength` is an array of **rest-frame wavelengths**,
         - `flux` is an array of flux values,
         - `ivar` is an array of inverse variance values,
         - `flags` is an optional array of flags.
    
    :param min_wavelength:
        The minimum rest wavelength to consider.
    
    :param max_wavelength:
        The maximum rest wavelength to consider.
    
    :param P:
        The number of Fourier modes to use.
    
    :param L:
        The length scale, usually taken as ``max_wavelength - min_wavelength``.

    :returns:
        A tuple of ``(A, YY, CI)``, where:
            - `A` is the design matrix,
            - `YY` is the flux vector,
            - `CI` is the inverse variance vector.
    """
    XX, YY, CI = ([], [], [])
    for wl, flux, ivar, *flags in visit_spectra:        
        if len(flags) > 0 and flags[0] is not None:
            keep = (flags[0] == 0)
            use_wl, use_flux, use_ivar = (wl[keep], flux[keep], ivar[keep])
        else:
            use_wl, use_flux, use_ivar = (wl, flux, ivar)

        I = np.logical_and(use_wl > min_wavelength, use_wl < max_wavelength)        
        XX.extend(use_wl[I])
        YY.extend(use_flux[I])
        CI.extend(use_ivar[I])
    A = construct_design_matrix(np.array(XX), P, L)
    YY, CI = (np.array(YY), np.array(CI))
    return (A, YY, CI)


def construct_design_matrix(wavelength: np.array, P: int, L: int):
    """
    Take in a set of wavelengths and return the Fourier design matrix.

    :param wavelength:
        An ``N``-length array of wavelength values.
    
    :param P:
        The number of Fourier modes to use.
    
    :param L:
        The length scale, usually taken as ``max(wavelength) - min(wavelength)``.
    
    :returns:
        A design matrix of shape (N, P).
    """
    # TODO: This could be replaced with something that makes use of finufft.
    scale = (np.pi * wavelength) / L
    X = np.ones((wavelength.size, P), dtype=float)
    for j, f in zip(range(1, P), cycle((np.sin, np.cos))):
        X[:, j] = f(scale * (j + (j % 2)))
    return X


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

    np.random.seed(17)
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
    print(Mstar, P, xstar.shape)

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
    xs2, ys2, y_ivars2, bs2, Delta_xs2 = make_one_dataset(dx2, SNR2)
    M2 = len(xs2)
    N2 = len(ys2)
    name2 = "well sampled input"
    print(name2, N2, M2, SNR2)
    ys_true = true_spectrum(xs2, 0)

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

    wavelength = xstar


    y, y_var = sample_combined_spectrum(wavelength, visit_spectra)


    fig, ax = plt.subplots()
    ax.plot(wavelength, y, c="k", drawstyle="steps-mid")
    ax.fill_between(
        wavelength,
        y - np.sqrt(y_var),
        y + np.sqrt(y_var),
        facecolor="#666666",
        zorder=-1,
        step="mid"
    )
    ax.plot(xs2, ys_true, c="tab:blue")
    ax.plot(xstar, y - true_spectrum(xstar, 0), c="tab:blue", drawstyle="steps-mid")

    for i in range(ys2.shape[0]):
        ax.plot(xs2 - Delta_xs2[i], ys2[i] + i + 1, c="k")
    fig.savefig("sampling.pdf")

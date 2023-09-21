# imports and initialize seeds, fundamental constants
import numpy as np
import pylab as plt
import scipy.interpolate as interp
np.random.seed(17)
c = 299792458. # m / s
sqrt2pi = np.sqrt(2. * np.pi)

# make the synthetic spectrum (spectral expectation), and also add noise

def oned_gaussian(dxs, sigma):
    return np.exp(-0.5 * dxs ** 2 / sigma ** 2) / (sqrt2pi * sigma)

def true_spectrum(xs, doppler, lxs, ews, sigma):
    """
    """
    return np.exp(-1. * np.sum(ews[None, :] *
                               oned_gaussian(xs[:, None] - doppler
                                             - lxs[None, :], sigma), axis=1))

def ivar(ys, continuum_ivar):
    return continuum_ivar / ys

def noisy_true_spectrum(xs, doppler, continuum_ivar, line_xs, line_ews, sigma_x):
    """
    """
    ys_true = true_spectrum(xs, doppler, line_xs, line_ews, sigma_x)
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

def badify(yy, badfrac):
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

def make_one_dataset(dx, SNR, x_min, x_max, line_xs, line_ews, sigma_x, badfrac, N=8):
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
        ys[j], y_ivars[j] = noisy_true_spectrum(xs, Delta_xs[j], continuum_ivar, line_xs, line_ews, sigma_x)
        bs[j], ys[j] = badify(ys[j], badfrac)
    return xs, ys, y_ivars, bs, Delta_xs



def test_poorly_sampled_regime():
    # define high-level parameters, especially including spectrograph parameters
    R = 1.35e5 # resolution
    sigma_x = 1. / R # LSF sigma in x units
    x_min = 8.7000 # minimum ln wavelength
    x_max = 8.7025 # maximum ln wavelength
    lines_per_x = 2.0e4 # mean density (Poisson rate) of lines per unit ln wavelength
    ew_max_x = 3.0e-5 # maximum equivalent width in x units
    ew_power = 5.0 # power parameter in EW maker
    badfrac = 0.01 # fraction of data to mark bad
    # Set the pixel grid and model complexity for the output combined spectrum
    dxstar = 1. / R # output pixel grid spacing
    xstar = np.arange(x_min + 0.5 * dxstar, x_max, dxstar) # output pixel grid
    Mstar = len(xstar) # number of output pixels
    P = np.round((x_max - x_min) * R).astype(int) # number of Fourier modes (ish)

    # set up the line list for the true spectral model
    x_margin = 1.e6/c # hoping no velocities are bigger than 1000 km/s
    x_range = x_max - x_min + 2. * x_margin # make lines in a bigger x range than the data range
    nlines = np.random.poisson(x_range * lines_per_x) # set the total number of lines
    line_xs = (x_min - x_margin) + x_range * np.random.uniform(size=nlines)

    # give those lines equivalent widths from a power-law distribution
    line_ews = ew_max_x * np.random.uniform(size=nlines) ** ew_power # don't ask

    dx1 = 2. / R # pixel spacing in the poorly sampled data; UNDER-SAMPLED!
    SNR1 = 18. # s/n ratio per pixel in the continuum
    xs1, ys1, y_ivars1, bs1, Delta_xs1 = make_one_dataset(dx1, SNR1, x_min, x_max, line_xs, line_ews, sigma_x, badfrac)
    M1 = len(xs1)
    N1 = len(ys1)
    name1 = "poorly sampled input"
    print(name1, N1, M1, SNR1)

    


def test_well_sampled_regime():

    R = 1.35e5 # resolution
    sigma_x = 1. / R # LSF sigma in x units
    x_min = 8.7000 # minimum ln wavelength
    x_max = 8.7025 # maximum ln wavelength
    lines_per_x = 2.0e4 # mean density (Poisson rate) of lines per unit ln wavelength
    ew_max_x = 3.0e-5 # maximum equivalent width in x units
    ew_power = 5.0 # power parameter in EW maker
    badfrac = 0.01 # fraction of data to mark bad
    # Set the pixel grid and model complexity for the output combined spectrum
    dxstar = 1. / R # output pixel grid spacing
    xstar = np.arange(x_min + 0.5 * dxstar, x_max, dxstar) # output pixel grid
    Mstar = len(xstar) # number of output pixels
    P = np.round((x_max - x_min) * R).astype(int) # number of Fourier modes (ish)

    # set up the line list for the true spectral model
    x_margin = 1.e6/c # hoping no velocities are bigger than 1000 km/s
    x_range = x_max - x_min + 2. * x_margin # make lines in a bigger x range than the data range
    nlines = np.random.poisson(x_range * lines_per_x) # set the total number of lines
    line_xs = (x_min - x_margin) + x_range * np.random.uniform(size=nlines)

    # give those lines equivalent widths from a power-law distribution
    line_ews = ew_max_x * np.random.uniform(size=nlines) ** ew_power # don't ask

    dx2 = 1. / R # pixel spacing in the poorly sampled data; UNDER-SAMPLED!
    SNR2 = 12. # s/n ratio per pixel in the continuum
    xs2, ys2, y_ivars2, bs2, Delta_xs2 = make_one_dataset(dx2, SNR2)
    M2 = len(xs2)
    N2 = len(ys2)
    name2 = "well sampled input"
    print(name2, N2, M2, SNR2)


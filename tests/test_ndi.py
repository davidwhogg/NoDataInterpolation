# imports and initialize seeds, fundamental constants
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt

np.random.seed(17)
c = 299792458. # m / s
sqrt2pi = np.sqrt(2. * np.pi)

from ndi import resample_spectrum

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
    return  (ys_true + np.random.normal(size=xs.shape) / np.sqrt(y_ivars), y_ivars)

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

def make_one_dataset(dx, SNR, x_min, x_max, line_xs, line_ews, sigma_x, badfrac, xstar, N=8):
    # create true Doppler shifts on a sinusoid of epoch number
    Delta_xs = (3.e4 / c) * np.cos(np.arange(N) / 3.)
    # set the ivar
    continuum_ivar = SNR ** 2 # inverse variance of the noise in the continuum
    # now make the noisy fake data
    xs = np.arange(x_min - 0.5 * dx, x_max + dx, dx)
    ys = np.zeros((N, len(xs)))
    y_ivars = np.zeros_like(ys)
    bs = np.zeros_like(ys).astype(int)
    y_true = true_spectrum(xstar, 0, line_xs, line_ews, sigma_x)
    for j in range(N):
        ys[j], y_ivars[j] = noisy_true_spectrum(xs, Delta_xs[j], continuum_ivar, line_xs, line_ews, sigma_x)
        bs[j], ys[j] = badify(ys[j], badfrac)
    return xs, ys, y_ivars, bs, Delta_xs, y_true


# Estimate covariances from just one trial:
def covariances(resids):
    lags = np.arange(12)
    var = np.zeros(len(lags)) + np.NaN
    var[0] = np.mean(resids * resids)
    for lag in lags[1:]:
        var[lag] = np.mean(resids[lag:] * resids[:-lag])
    return lags, var

def get_poorly_sampled_regime_data():
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
    xs1, ys1, y_ivars1, bs1, Delta_xs1, y_true = make_one_dataset(dx1, SNR1, x_min, x_max, line_xs, line_ews, sigma_x, badfrac, xstar)

    return (dx1, SNR1, xs1, ys1, y_ivars1, bs1, Delta_xs1, xstar, y_true, line_xs, line_ews, sigma_x)




def test_poorly_sampled_regime():
   
    (dx1, SNR1, xs1, ys1, y_ivars1, bs1, Delta_xs1, xstar, y_true, line_xs, line_ews, sigma_x) = get_poorly_sampled_regime_data()

    M1 = len(xs1)
    N1 = len(ys1)
    name1 = "poorly sampled input"
    print(name1, N1, M1, SNR1)

    x = []
    for delta_xs in Delta_xs1:
        x.extend(xs1 - delta_xs)

    y_star, Cinv_star, _ = resample_spectrum(xstar, x, ys1.flatten(), y_ivars1.flatten(), mask=(bs1.flatten() != 1))

    z = (y_star - y_true) * np.sqrt(Cinv_star)


    fig, ax = plt.subplots()

    ax.plot(xstar, y_star, c='k')
    ax.plot(xstar, y_true, c="tab:blue")
    ax.plot(xstar, y_star - y_true, c="k")


    for i, xs in enumerate(Delta_xs1):
        ax.plot(xs1 - xs, ys1[i] + i + 1, c='#666666')
        
    assert np.abs(np.mean(z)) < 0.15
    assert np.abs(np.std(z) - 1) < 0.05

    return None




def get_well_sampled_regime_data():
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
    xs2, ys2, y_ivars2, bs2, Delta_xs2, y_true = make_one_dataset(dx2, SNR2,  x_min, x_max, line_xs, line_ews, sigma_x, badfrac, xstar)

    return (dx2, SNR2, xs2, ys2, y_ivars2, bs2, Delta_xs2, xstar, y_true, line_xs, line_ews, sigma_x)


def test_well_sampled_regime():

    (dx2, SNR2, xs2, ys2, y_ivars2, bs2, Delta_xs2, xstar, y_true, line_xs, line_ews, sigma_x) = get_well_sampled_regime_data()

    M2 = len(xs2)
    N2 = len(ys2)
    name2 = "well sampled input"
    print(name2, N2, M2, SNR2)

    x = []
    for delta_xs in Delta_xs2:
        x.extend(xs2 - delta_xs)

    y_star, Cinv_star, _ = resample_spectrum(xstar, x, ys2.flatten(), y_ivars2.flatten(), mask=(bs2.flatten() != 1))


    z = (y_star - y_true) * np.sqrt(Cinv_star)


    fig, ax = plt.subplots()

    ax.plot(xstar, y_star, c='k')
    ax.plot(xstar, y_true, c="tab:blue")
    ax.plot(xstar, y_star - y_true, c="k")


    for i, xs in enumerate(Delta_xs2):
        ax.plot(xs2 - xs, ys2[i] + i + 1, c='#666666')
    
    assert np.abs(np.mean(z)) < 0.05
    assert np.abs(np.std(z) - 1) < 0.05

    return fig


def Standard_Practice_tm(xs, ys, bs, Delta_xs, xstar, kind="cubic"):
    #interpolate the data and the masks; deal with edges.
    # Note that we are being very conservative with the mask.
    N = len(ys)
    yprimes = np.zeros((N, len(xstar)))
    bprimes = np.zeros_like(yprimes).astype(int)
    ikwargs = {"kind": kind, "fill_value": "extrapolate"}
    for j in range(N):
        yprimes[j] = interp.interp1d(xs - Delta_xs[j], ys[j],
                                     **ikwargs)(xstar)
        bprimes[j] = (np.abs(interp.interp1d(xs - Delta_xs[j], bs[j],
                                     **ikwargs)(xstar) - 1.) < 0.03).astype(int)
        bprimes[j][xstar < (min(xs) - Delta_xs[j])] = 0
        bprimes[j][xstar > (max(xs) - Delta_xs[j])] = 0
    ystar = np.sum(yprimes * bprimes, axis=0) / np.sum(bprimes, axis=0)
    return ystar, yprimes, bprimes


def test_lag():
    # Estimate using multiple repeated experiments
    #(dx1, SNR1, xs1, ys1, y_ivars1, bs1, Delta_xs1, xstar, y_true1, line_xs, line_ews, sigma_x) = get_poorly_sampled_regime_data()
    #(dx2, SNR2, xs2, ys2, y_ivars2, bs2, Delta_xs2, xstar, y_true2, line_xs, line_ews, sigma_x) = get_well_sampled_regime_data()
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
    dx2 = 1. / R # pixel spacing in the poorly sampled data; UNDER-SAMPLED!
    SNR2 = 12. # s/n ratio per pixel in the continuum

    name1 = "poorly sampled input"
    name2 = "well sampled input"

    ntrial = 64
    for i, dx, SNR in [(1, dx1, SNR1),
                    (2, dx2, SNR2)]:
        numerator = 0.
        numerator_sp = 0.
        for trial in range(ntrial):
            xs, ys, y_ivars, bs, Delta_xs, y_true = make_one_dataset(dx, SNR, x_min, x_max, line_xs, line_ews, sigma_x, badfrac, xstar)
            x = []
            for delta_xs in Delta_xs:
                x.extend(xs - delta_xs)

            y_star, Cinv_star, _ = resample_spectrum(xstar, x, ys.flatten(), y_ivars.flatten(), mask=(bs.flatten() != 1))
            ystar_sp, foo, bar = Standard_Practice_tm(xs, ys, bs, Delta_xs, xstar)
            lags, covars = covariances(y_star - y_true)
            # sometimes standard practice will have nans at the edge
            finite_sp = ystar_sp - y_true
            lags, covars_sp = covariances((ystar_sp - y_true)[np.isfinite(finite_sp)])
            assert np.all(np.isfinite(covars_sp))
            assert np.all(np.isfinite(covars))

            numerator += covars
            numerator_sp += covars_sp

        if i == 1:
            covars1 = numerator / ntrial
            covars_sp1 = numerator_sp / ntrial
        elif i == 2:
            covars2 = numerator / ntrial
            covars_sp2 = numerator_sp / ntrial    

    # the forward model should have a lower mean covariance
    assert np.mean(np.abs(covars1)[1:]) < np.mean(np.abs(covars_sp1)[1:])
    assert np.mean(np.abs(covars2)[1:]) < np.mean(np.abs(covars_sp2)[1:])


    fig, ax = plt.subplots()
    ax.axhline(0., color="k", lw=0.5)
    ax.plot(lags, covars1, "ko", ms=5,
            label="Forward Model (tm), " + name1)
    ax.plot(lags, covars_sp1, "ko", ms=5, mfc="none",
            label="Standard Practice (tm), " + name1)
    ax.plot(lags, covars2, "ko", ms=10, alpha=0.5, mec="none",
            label="Forward Model (tm), " + name2)
    ax.plot(lags, covars_sp2, "ko", ms=10, alpha=0.5, mfc="none",
            label="Standard Practice (tm), " + name2)
    fig.legend()
    ax.set_xlabel("lag (in output pixels)")
    ax.set_ylabel("covariance (squared-flux units)")
    ax.set_title("covariances estimated from {} trials".format(ntrial))
    return None


'''
if __name__ == "__main__":

    fig = test_lag()
    fig_well_sampled = test_well_sampled_regime()
    fig_poorly_sampled = test_poorly_sampled_regime()

'''

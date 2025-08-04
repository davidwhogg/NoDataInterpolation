import numpy as np
from scipy.special import wofz

# Global constants
c = 299792458.  # m / s
sqrt2pi = np.sqrt(2. * np.pi)

def oned_gaussian(dxs, sigma):
    """1D Gaussian profile function"""
    return np.exp(-0.5 * dxs ** 2 / sigma ** 2) / (sqrt2pi * sigma)

def oned_lorentzian(dxs, gamma):
    """1D Lorentzian profile function"""
    return gamma / (np.pi * (dxs ** 2 + gamma ** 2))

def oned_voigt(dxs, sigma, gamma):
    """1D Voigt profile function (convolution of Gaussian and Lorentzian)"""
    z = (dxs + 1j * gamma) / (sigma * np.sqrt(2))
    return np.real(wofz(z)) / (sigma * sqrt2pi * np.sqrt(2))

def get_profile_function(profile_type, profile_width):
    """
    Returns the appropriate profile function based on type
    
    Parameters:
    -----------
    profile_type : str
        'gaussian', 'lorentzian', or 'voigt'
    profile_width : float or tuple
        For gaussian: sigma
        For lorentzian: gamma
        For voigt: (sigma, gamma) tuple
    
    Returns:
    --------
    function : callable profile function
    """
    if profile_type.lower() == 'gaussian':
        return lambda dxs: oned_gaussian(dxs, profile_width)
    elif profile_type.lower() == 'lorentzian':
        return lambda dxs: oned_lorentzian(dxs, profile_width)
    elif profile_type.lower() == 'voigt':
        if not isinstance(profile_width, (tuple, list)) or len(profile_width) != 2:
            raise ValueError("For Voigt profile, profile_width must be (sigma, gamma) tuple")
        sigma, gamma = profile_width
        return lambda dxs: oned_voigt(dxs, sigma, gamma)
    else:
        raise ValueError("profile_type must be 'gaussian', 'lorentzian', or 'voigt'")

def true_spectrum(xs, doppler, line_xs, line_ews, profile_func):
    """
    Calculate the true spectrum given line positions, equivalent widths, and profile function
    """
    return np.exp(-1. * np.sum(line_ews[None, :] *
                               profile_func(xs[:, None] - doppler
                                           - line_xs[None, :]), axis=1))

def ivar(ys, continuum_ivar):
    """Calculate inverse variance"""
    return continuum_ivar / ys

def noisy_true_spectrum(xs, doppler, line_xs, line_ews, profile_func, continuum_ivar):
    """
    Generate noisy spectrum
    """
    ys_true = true_spectrum(xs, doppler, line_xs, line_ews, profile_func)
    y_ivars = ivar(ys_true, continuum_ivar)
    ys_noisy = ys_true + np.random.normal(size=xs.shape) / np.sqrt(y_ivars)
    return (ys_noisy, y_ivars)

def doppler_information(xs, doppler, line_xs, line_ews, profile_func, continuum_ivar, dx):
    """
    Calculate Doppler information
    # Bugs:
    - Horrifying numerical derivative!
    """
    dys_dx = (true_spectrum(xs, doppler + dx, line_xs, line_ews, profile_func)
              - true_spectrum(xs, doppler - dx, line_xs, line_ews, profile_func)) / (2. * dx)
    y_ivars = ivar(true_spectrum(xs, doppler, line_xs, line_ews, profile_func), continuum_ivar)
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

def make_one_dataset(dx, snr, N=8, 
                    x_min=8.7000, x_max=8.7025, R=1.35e5,
                    lines_per_x=2.0e4, ew_max_x=3.0e-5, ew_power=5.0,
                    badfrac=0.01, profile_type='gaussian', profile_width=None,
                    random_seed=None, Delta_xs=None):
    """
    Generate one synthetic dataset with specified parameters
    
    Parameters:
    -----------
    dx : float
        Pixel spacing
    snr : float
        Signal-to-noise ratio per pixel in the continuum
    N : int, optional
        Number of epochs (default: 8)
    x_min : float, optional
        Minimum ln wavelength (default: 8.7000)
    x_max : float, optional  
        Maximum ln wavelength (default: 8.7025)
    R : float, optional
        Spectral resolution (default: 1.35e5)
    lines_per_x : float, optional
        Mean density of lines per unit ln wavelength (default: 2.0e4)
    ew_max_x : float, optional
        Maximum equivalent width in x units (default: 3.0e-5)
    ew_power : float, optional
        Power parameter in EW distribution (default: 5.0)
    badfrac : float, optional
        Fraction of data to mark bad (default: 0.01)
    profile_type : str, optional
        Profile function type: 'gaussian', 'lorentzian', or 'voigt' (default: 'gaussian')
    profile_width : float or tuple, optional
        Profile width parameter(s). If None, uses sigma_x = 1/R for gaussian,
        gamma = 1/R for lorentzian, or (1/R, 1/R) for voigt
    random_seed : int, optional
        Random seed for reproducibility. If None, uses existing random state
        
    Returns:
    --------
    xs : array
        Wavelength grid
    ys : array
        Observed spectra (N x M)
    y_ivars : array  
        Inverse variances (N x M)
    bs : array
        Bad pixel masks (N x M)
    Delta_xs : array
        True Doppler shifts (N,)
    """
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Set default profile width if not provided
    if profile_width is None:
        sigma_x = 1. / R
        if profile_type.lower() == 'gaussian':
            profile_width = sigma_x
        elif profile_type.lower() == 'lorentzian':
            profile_width = sigma_x  # Use same width for consistency
        elif profile_type.lower() == 'voigt':
            profile_width = (sigma_x, sigma_x)  # (sigma, gamma)
    
    # Get the profile function
    profile_func = get_profile_function(profile_type, profile_width)
    
    # Set up the line list for the true spectral model
    x_margin = 1.e6/c  # hoping no velocities are bigger than 1000 km/s
    x_range = x_max - x_min + 2. * x_margin  # make lines in a bigger x range than the data range
    nlines = np.random.poisson(x_range * lines_per_x)  # set the total number of lines
    line_xs = (x_min - x_margin) + x_range * np.random.uniform(size=nlines)
    
    # Give those lines equivalent widths from a power-law distribution
    line_ews = ew_max_x * np.random.uniform(size=nlines) ** ew_power
    
    # Create true Doppler shifts on a sinusoid of epoch number
    if Delta_xs is None:
        Delta_xs = (3.e4 / c) * np.cos(np.arange(N) / 3.)
    
    # Set the ivar
    continuum_ivar = snr ** 2  # inverse variance of the noise in the continuum
    
    # Now make the noisy fake data
    xs = np.arange(x_min - 0.5 * dx, x_max + dx, dx)
    ys = np.zeros((N, len(xs)))
    y_ivars = np.zeros_like(ys)
    bs = np.zeros_like(ys).astype(int)
    
    for j in range(N):
        ys[j], y_ivars[j] = noisy_true_spectrum(xs, Delta_xs[j], line_xs, line_ews, 
                                               profile_func, continuum_ivar)
        bs[j], ys[j] = badify(ys[j], badfrac)
    
    return xs, ys, y_ivars, bs, Delta_xs, (line_xs, line_ews, profile_func)


"""my_xs2, my_ys2, my_y_ivars2, my_bs2, my_Delta_xs2 = make_one_dataset(
        dx2, snr2, N=8, x_min=x_min, x_max=x_max, R=R,
        lines_per_x=lines_per_x, ew_max_x=ew_max_x, ew_power=ew_power,
        badfrac=badfrac, profile_type='gaussian', random_seed=17
)

assert np.allclose(my_xs2, xs2)
assert np.allclose(my_ys2, ys2)
assert np.allclose(my_y_ivars2, y_ivars2)
assert np.allclose(my_bs2, bs2)
assert np.allclose(my_Delta_xs2, Delta_xs2)


my_xs1, my_ys1, my_y_ivars1, my_bs1, my_Delta_xs1 = make_one_dataset(
    dx1, snr1, N=8, x_min=x_min, x_max=x_max, R=R,
    lines_per_x=lines_per_x, ew_max_x=ew_max_x, ew_power=ew_power,
    badfrac=badfrac, profile_type='gaussian', random_seed=17
)

assert np.allclose(my_xs1, xs1)
assert np.allclose(my_ys1, ys1)
assert np.allclose(my_y_ivars1, y_ivars1)
assert np.allclose(my_bs1, bs1)
assert np.allclose(my_Delta_xs1, Delta_xs1)
"""


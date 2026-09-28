"""Lorentzian fit of the exact transmission spectrum around a resonance."""
import numpy as np
from scipy.optimize import curve_fit

from .tmm import transmission_reflection


def lorentzian(k, k_c, gamma, amplitude, offset):
    return amplitude / ((k - k_c) ** 2 + gamma**2) + offset


def fit_resonance(layers, k_center, gamma_guess, n_b=1.0, half_window=3.0, npts=401):
    """Fit |t(k)|^2 on k_center +/- half_window * gamma_guess.

    Returns the fitted complex wavenumber k_c - i gamma, in the same convention as the
    quasinormal-mode eigenvalue.
    """
    ks = np.linspace(k_center - half_window * gamma_guess, k_center + half_window * gamma_guess, npts)
    power = np.array([abs(transmission_reflection(k, layers, n_b)[0]) ** 2 for k in ks])
    p0 = (k_center, gamma_guess, power.max() * gamma_guess**2, 0.0)
    popt, _ = curve_fit(lorentzian, ks, power, p0=p0, maxfev=20000)
    return complex(popt[0], -abs(popt[1]))

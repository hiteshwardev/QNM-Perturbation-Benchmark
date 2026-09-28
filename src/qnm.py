"""Closed-form quasinormal modes of a single dielectric slab.

The slab of index n_r occupies [x1, x1 + L] in a background of index n_b. With time
dependence exp(-i omega t) the outgoing boundary conditions give

    n_r k_mu L = mu pi + i ln[(n_r - n_b) / (n_r + n_b)],

so every mode has the same imaginary part and decays in time. Outside the slab the mode
continues as an outgoing plane wave with the complex wavenumber, which makes |f(x)|^2 grow
exponentially with distance from the slab.
"""
import numpy as np


def slab_wavenumber(mu, n_r, length=1.0, n_b=1.0):
    """Quasinormal-mode wavenumber k_mu in units of 1/L."""
    return (mu * np.pi + 1j * np.log((n_r - n_b) / (n_r + n_b))) / (n_r * length)


def slab_mode(mu, n_r, length=1.0, n_b=1.0, x1=-1.0):
    """Normalized mode of index mu.

    Returns (f, k, f_left, f_right): f(x, k_ext=None) evaluates the field anywhere, using
    the exterior wavenumber k_ext outside the slab when given (the mode's own k otherwise);
    f_left and f_right are the field values at the two slab faces. The amplitude satisfies
    the normalization of Leung et al., Phys. Rev. A 49, 3068 (1994), under which the
    first-order shift of k is -(k/2) times the overlap of the permittivity change with f^2.
    """
    k = slab_wavenumber(mu, n_r, length, n_b)
    x2 = x1 + length
    xc = x1 + 0.5 * length
    parity = (-1.0) ** mu
    amp = 1.0 / (np.sqrt(parity * 2.0 * length) * n_r)
    f_right = amp * (np.exp(1j * n_r * k * (x2 - xc)) + parity * np.exp(-1j * n_r * k * (x2 - xc)))
    f_left = amp * (np.exp(1j * n_r * k * (x1 - xc)) + parity * np.exp(-1j * n_r * k * (x1 - xc)))

    def f(x, k_ext=None):
        ke = k if k_ext is None else k_ext
        x = np.asarray(x, dtype=float)
        inside = amp * (np.exp(1j * n_r * k * (x - xc)) + parity * np.exp(-1j * n_r * k * (x - xc)))
        right = f_right * np.exp(1j * n_b * ke * (x - x2))
        left = f_left * np.exp(-1j * n_b * ke * (x - x1))
        return np.where(x > x2, right, np.where(x < x1, left, inside))

    return f, k, f_left, f_right

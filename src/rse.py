"""Resonant-state expansion for planar structures with a homogeneous basis slab.

Follows Doost, Langbein and Muljarov, Phys. Rev. A 85, 023835 (2012). The basis system is
a slab of permittivity eps_s on [-a, a] in vacuum; its resonant states are known in closed
form. Any structure contained in [-a, a] is written as the basis slab plus a piecewise-
constant change of permittivity, and the perturbed wavenumbers follow from the linear
eigenproblem

    (1 / k) c_n = sum_m [delta_nm / k_n + V_nm / (2 k_m)] c_m,

a similarity transform of the symmetric form of that paper. Units: c = 1.
"""
import numpy as np


def basis_wavenumbers(n, eps_s, a):
    """k_n = [pi n - i ln((sqrt(eps_s) + 1) / (sqrt(eps_s) - 1))] / (2 a sqrt(eps_s))."""
    q = np.sqrt(eps_s)
    return (np.pi * n - 1j * np.log((q + 1.0) / (q - 1.0))) / (2.0 * a * q)


def basis_amplitudes(n, eps_s, a):
    return ((-1j) ** n) / (2.0 * np.sqrt(a * eps_s))


def perturbation_matrix(nmax, eps_s, a, pieces):
    """Matrix elements V_nm = integral of delta_eps(z) E_n(z) E_m(z) over [-a, a].

    pieces is a list of (z1, z2, delta_eps) with constant delta_eps on [z1, z2]; the
    integrals are evaluated analytically. Returns the basis indices and V.
    """
    ns = np.arange(-nmax, nmax + 1)
    k = basis_wavenumbers(ns, eps_s, a)
    amp = basis_amplitudes(ns, eps_s, a)
    parity = (-1.0) ** ns
    q = np.sqrt(eps_s)
    kp = q * (k[:, None] + k[None, :])
    km = q * (k[:, None] - k[None, :])

    def exp_integral(kappa, z1, z2):
        small = np.abs(kappa) < 1e-14
        safe = np.where(small, 1.0, kappa)
        value = (np.exp(1j * safe * z2) - np.exp(1j * safe * z1)) / (1j * safe)
        return np.where(small, z2 - z1, value)

    v = np.zeros((ns.size, ns.size), dtype=complex)
    for z1, z2, de in pieces:
        v += de * (amp[:, None] * amp[None, :]) * (
            exp_integral(kp, z1, z2)
            + parity[None, :] * exp_integral(km, z1, z2)
            + parity[:, None] * exp_integral(-km, z1, z2)
            + parity[:, None] * parity[None, :] * exp_integral(-kp, z1, z2))
    return ns, v


def rse_wavenumbers(nmax, eps_s, a, pieces):
    """Perturbed resonant-state wavenumbers for a basis of N = 2 nmax + 1 states."""
    ns, v = perturbation_matrix(nmax, eps_s, a, pieces)
    k = basis_wavenumbers(ns, eps_s, a)
    matrix = np.diag(1.0 / k) + v / (2.0 * k[None, :])
    return 1.0 / np.linalg.eigvals(matrix)


def nearest(values, target):
    """Element of values closest to target in the complex plane."""
    values = np.asarray(values)
    return values[np.argmin(np.abs(values - target))]

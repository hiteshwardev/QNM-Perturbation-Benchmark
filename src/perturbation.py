"""Perturbative estimates of the shift of a slab quasinormal mode caused by a thin layer of
permittivity change delta_eps on [d, d + lp], outside the cavity [-L, 0].

Exterior overlap. With the exterior field continued from the right cavity face at the
wavenumber k_x, the first-order overlap is

    F(k_x) = -(k_0 / 2) delta_eps f_R^2 exp(2 i n_b k_x d) [exp(2 i n_b k_x lp) - 1] / (2 i n_b k_x),

where f_R is the mode amplitude at the right cavity face.

Pole function. The exact perturbed pole satisfies 1 = rho_c(k) rho_p(k) exp(2 i n_b k d), with
rho_c the reflection coefficient of the cavity seen from the gap (reference plane x = 0) and
rho_p that of the layer (reference plane x = d). Splitting rho_c = R / (k - k_0) + rho_bg(k),
with residue R = i n_b f_R^2, gives

    k - k_0 = G(k) / [1 - alpha G(k)],   G(k) = R rho_p(k) exp(2 i n_b k d),   alpha = rho_bg / R.

With rho_p replaced by its first Born approximation, G(k) = (k / k_0) F(k).

Schemes compared in this study:
    naive            dk = F(k_0)
    regularized      dk = F(Re k_0)
    self-consistent  dk = G(k_0 + dk)
    resummed         dk = G(k_0 + dk) / [1 - alpha G(k_0 + dk)]

The resummed scheme can also be evaluated with the second Born approximation of rho_p.
"""
import numpy as np

from .qnm import slab_mode, slab_wavenumber


def exterior_overlap(k_x, mu, d, delta_eps, n_r, length=1.0, lp=0.1, n_b=1.0):
    """First-order overlap F(k_x) with the exterior field evaluated at wavenumber k_x."""
    _, k0, _, f_right = slab_mode(mu, n_r, length, n_b, x1=-length)
    integral = (f_right**2 * np.exp(2j * n_b * k_x * d)
                * (np.exp(2j * n_b * k_x * lp) - 1.0) / (2j * n_b * k_x))
    return -(k0 / 2.0) * delta_eps * integral


def naive_shift(mu, d, delta_eps, n_r, length=1.0, lp=0.1, n_b=1.0):
    k0 = slab_wavenumber(mu, n_r, length, n_b)
    return exterior_overlap(k0, mu, d, delta_eps, n_r, length, lp, n_b)


def regularized_shift(mu, d, delta_eps, n_r, length=1.0, lp=0.1, n_b=1.0):
    k0 = slab_wavenumber(mu, n_r, length, n_b)
    return exterior_overlap(k0.real, mu, d, delta_eps, n_r, length, lp, n_b)


def pole_function(k, mu, d, delta_eps, n_r, length=1.0, lp=0.1, n_b=1.0, born_order=1):
    """G(k) and dG/dk for the Born approximation of the layer reflection.

    born_order=1 uses rho_p = delta_eps [exp(2 i n_b k lp) - 1] / (4 n_b^2). born_order=2 adds
    the second Born term, which multiplies G by
    1 + delta_eps [i k lp E / (n_b (E - 1)) - 1 / (2 n_b^2)], with E = exp(2 i n_b k lp).
    """
    _, _, _, f_right = slab_mode(mu, n_r, length, n_b, x1=-length)
    e_d = np.exp(2j * n_b * k * d)
    e_p = np.exp(2j * n_b * k * lp)
    value = 1j * delta_eps * f_right**2 * e_d * (e_p - 1.0) / (4.0 * n_b)
    derivative = value * (2j * n_b * d + 2j * n_b * lp * e_p / (e_p - 1.0))
    if born_order == 2:
        h = 1.0 + delta_eps * (1j * k * lp * e_p / (n_b * (e_p - 1.0)) - 0.5 / n_b**2)
        dh = delta_eps * (1j * lp / n_b) * (e_p / (e_p - 1.0)
                                            - 2j * n_b * lp * k * e_p / (e_p - 1.0) ** 2)
        value, derivative = value * h, derivative * h + value * dh
    elif born_order != 1:
        raise ValueError("born_order must be 1 or 2")
    return value, derivative


def slab_background_ratio(n_r, length=1.0, n_b=1.0):
    """alpha = rho_bg / R for a slab cavity: i n_r L (1 + r^2) / (1 - r^2), r = (n_r - n_b) / (n_r + n_b).

    The slab reflection depends on k only through exp(2 i n_r k L), so alpha is the same for
    every mode of the slab.
    """
    r = (n_r - n_b) / (n_r + n_b)
    return 1j * n_r * length * (1.0 + r**2) / (1.0 - r**2)


def second_order_terms(d, mu, n_r, alpha, length=1.0, lp=0.1, n_b=1.0):
    """The four terms of a(d) in dk_exact = F(k_0) + a(d) F(k_0)^2 + O(delta_eps^3).

    Returns a dict with the propagation term 2 i n_b d, the layer term 2 i n_b lp E / (E - 1),
    the background term alpha, and the second Born term b exp(-2 i n_b k_0 d), where
    E = exp(2 i n_b k_0 lp) and b = [4 i n_b k_0 lp E / (E - 1)^2 - 2 / (E - 1)] / R.
    """
    _, k0, _, f_right = slab_mode(mu, n_r, length, n_b, x1=-length)
    residue = 1j * n_b * f_right**2
    e = np.exp(2j * n_b * k0 * lp)
    b = (4j * n_b * k0 * lp * e / (e - 1.0) ** 2 - 2.0 / (e - 1.0)) / residue
    d = np.asarray(d, dtype=float)
    return {"propagation": 2j * n_b * d,
            "layer": np.full(d.shape, 2j * n_b * lp * e / (e - 1.0)),
            "background": np.full(d.shape, complex(alpha)),
            "born": b * np.exp(-2j * n_b * k0 * d)}


def second_order_coefficient(d, mu, n_r, alpha, length=1.0, lp=0.1, n_b=1.0):
    """Coefficient a(d), the sum of the four terms returned by second_order_terms."""
    terms = second_order_terms(d, mu, n_r, alpha, length, lp, n_b)
    return terms["propagation"] + terms["layer"] + terms["background"] + terms["born"]


def solve_shift(mu, d, delta_eps, n_r, alpha=0.0, guess=None, length=1.0, lp=0.1, n_b=1.0,
                born_order=1, maxit=100, tol=1e-12):
    """Solve dk = G(k_0 + dk) / [1 - alpha G(k_0 + dk)] by complex Newton iteration.

    alpha = 0 gives the self-consistent scheme. The iteration starts from guess (the naive
    shift if None), is confined to |dk| < |k_0| / 2, and limits each step to a quarter of
    that radius. born_order selects the Born approximation of the layer reflection used in
    G. The iteration stops when a Newton step is smaller than tol |dk|; the default leaves room
    for the rounding error of exp(2 i n_b k d) at large gaps. Returns (dk, converged); dk is
    NaN when the iteration fails.
    """
    k0 = slab_wavenumber(mu, n_r, length, n_b)
    radius = 0.5 * abs(k0)
    dk = complex(pole_function(k0, mu, d, delta_eps, n_r, length, lp, n_b)[0]
                 if guess is None else guess)
    converged = False
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(maxit):
            if not np.isfinite(dk) or abs(dk) > radius:
                break
            g, dg = pole_function(k0 + dk, mu, d, delta_eps, n_r, length, lp, n_b, born_order)
            denominator = 1.0 - alpha * g
            residual = dk - g / denominator
            slope = 1.0 - dg / denominator**2
            if not (np.isfinite(residual) and np.isfinite(slope)) or slope == 0:
                break
            step = residual / slope
            if abs(step) > 0.25 * radius:
                step *= 0.25 * radius / abs(step)
            dk -= step
            if abs(step) <= tol * max(abs(dk), 1e-30):
                converged = True
                break
    if not converged or not np.isfinite(dk) or abs(dk) > radius:
        return complex(np.nan, np.nan), False
    return dk, True


def regularization_floor(d, mu, n_r, length=1.0, lp=0.1, n_b=1.0, thin_layer=False):
    """Small-strength relative error of the regularized scheme, |1 - c exp(2 n_b Im(k_0) d)|.

    c = k_0 [exp(2 i n_b Re(k_0) lp) - 1] / (Re(k_0) [exp(2 i n_b k_0 lp) - 1]) accounts for the
    finite layer thickness; thin_layer=True sets c = 1.
    """
    k0 = slab_wavenumber(mu, n_r, length, n_b)
    c = 1.0 if thin_layer else (k0 * (np.exp(2j * n_b * k0.real * lp) - 1.0)
                                / (k0.real * (np.exp(2j * n_b * k0 * lp) - 1.0)))
    return np.abs(1.0 - c * np.exp(2.0 * n_b * k0.imag * np.asarray(d, dtype=float)))

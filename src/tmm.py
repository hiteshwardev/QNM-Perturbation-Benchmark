"""Transfer matrices and pole finding for piecewise-constant one-dimensional structures.

In layer j the field is A_j exp(i k n_j x) + B_j exp(-i k n_j x), and the field and its
derivative are continuous at every interface. Units: c = 1, lengths in units of the cavity
length L, wavenumbers in units of 1/L. Time dependence exp(-i omega t).
"""
import cmath

import numpy as np


def _product(k, layers, n_b):
    """Entries (m11, m12, m21, m22) of the transfer matrix of layers [(n_1, t_1), ...] in index n_b.

    The matrix is the ordered product of interface matrices [[1, r], [r, 1]] / t, with
    r = (n_1 - n_2) / (n_1 + n_2) and t = 2 n_1 / (n_1 + n_2), and propagation matrices
    diag(exp(-i n k t_j), exp(i n k t_j)), evaluated in scalar complex arithmetic.
    """
    m11, m12, m21, m22 = 1.0 + 0j, 0j, 0j, 1.0 + 0j
    n_prev = n_b
    for n, thickness in list(layers) + [(n_b, None)]:
        r = (n_prev - n) / (n_prev + n)
        inv_t = (n_prev + n) / (2.0 * n_prev)
        a11, a12, a21, a22 = m11 + m12 * r, m11 * r + m12, m21 + m22 * r, m21 * r + m22
        m11, m12, m21, m22 = a11 * inv_t, a12 * inv_t, a21 * inv_t, a22 * inv_t
        if thickness is not None:
            phase = cmath.exp(1j * n * k * thickness)
            m11, m12, m21, m22 = m11 / phase, m12 * phase, m21 / phase, m22 * phase
        n_prev = n
    return m11, m12, m21, m22


def dispersion(k, layers, n_b=1.0):
    """D(k) = M_11(k). Its complex zeros are the quasinormal-mode wavenumbers."""
    return _product(complex(k), layers, n_b)[0]


def transmission_reflection(k, layers, n_b=1.0):
    """Complex amplitude transmission and reflection coefficients for incidence from the left."""
    m11, _, m21, _ = _product(complex(k), layers, n_b)
    return 1.0 / m11, m21 / m11


def double_barrier(d, n_r, delta_eps, length=1.0, lp=0.1, n_b=1.0):
    """Cavity of index n_r on [-L, 0] and a layer of relative permittivity n_b**2 + delta_eps
    on [d, d + lp]; the gap has the background index."""
    n_p = cmath.sqrt(n_b**2 + delta_eps)
    return [(n_r, length), (n_b, d), (n_p, lp)]


def find_pole(fun, k_start, tol=1e-12, maxit=80):
    """Complex Newton iteration with a central-difference derivative.

    Returns k with |fun(k)| < tol, refined by one further Newton step when that step lowers
    |fun(k)|; raises RuntimeError otherwise. Steps longer than half the current |k| are
    shortened to keep the iteration near the starting point.
    """
    k = complex(k_start)
    for _ in range(maxit):
        f = fun(k)
        h = 1e-7 * max(1.0, abs(k))
        if abs(f) < tol:
            k_new = k - f / ((fun(k + h) - fun(k - h)) / (2.0 * h))
            return k_new if abs(fun(k_new)) <= abs(f) else k
        step = f / ((fun(k + h) - fun(k - h)) / (2.0 * h))
        limit = 0.5 * max(1.0, abs(k))
        if abs(step) > limit:
            step *= limit / abs(step)
        k -= step
    raise RuntimeError(f"no convergence from k = {k_start}: |D| = {abs(f):.3e}")


def track_pole(fun, params, k_start, tol=1e-12):
    """Fixed-step continuation: each root seeds the search at the next parameter value."""
    roots = np.empty(len(params), dtype=complex)
    k = complex(k_start)
    for i, p in enumerate(params):
        k = find_pole(lambda kk: fun(kk, p), k, tol=tol)
        roots[i] = k
    return roots


def track_pole_adaptive(fun, params, k_start, max_step, tol=1e-12, max_depth=14):
    """Continuation with recursive bisection of the parameter step.

    A step is accepted when the root moves by at most max_step; otherwise the parameter
    interval is halved, up to max_depth times. Returns the roots and a flag per grid point
    that is False where a step could not be reduced below max_step.
    """
    def advance(k_from, p_from, p_to, depth):
        k_to = find_pole(lambda kk: fun(kk, p_to), k_from, tol=tol)
        if abs(k_to - k_from) <= max_step or depth >= max_depth:
            return k_to, abs(k_to - k_from) <= max_step
        p_mid = 0.5 * (p_from + p_to)
        k_mid, ok_1 = advance(k_from, p_from, p_mid, depth + 1)
        k_end, ok_2 = advance(k_mid, p_mid, p_to, depth + 1)
        return k_end, ok_1 and ok_2

    roots = np.empty(len(params), dtype=complex)
    ok = np.ones(len(params), dtype=bool)
    k = find_pole(lambda kk: fun(kk, params[0]), k_start, tol=tol)
    roots[0] = k
    for i in range(1, len(params)):
        k, ok[i] = advance(k, params[i - 1], params[i], 0)
        roots[i] = k
    return roots, ok

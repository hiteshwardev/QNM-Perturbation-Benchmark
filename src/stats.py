"""Straight-line fits with autocorrelation-robust uncertainties."""
import numpy as np


def hac_linfit(x, y, lag=None):
    """Least-squares line y = a x + b with a Newey-West (Bartlett kernel) standard error on a.

    Residuals of the growth-rate fits in this study oscillate with the cavity round-trip
    phase and are serially correlated, so the ordinary least-squares error understates the
    uncertainty. Returns (slope, intercept, sigma_hac, sigma_ols). The default lag is
    floor(4 (n / 100)^(2/9)), at least 1.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    design = np.column_stack([x, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    bread = np.linalg.inv(design.T @ design)
    dof = max(n - 2, 1)
    sigma_ols = np.sqrt(resid @ resid / dof * bread[0, 0])
    if lag is None:
        lag = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    scores = design * resid[:, None]
    meat = scores.T @ scores
    for j in range(1, min(lag, n - 1) + 1):
        cross = scores[j:].T @ scores[:-j]
        meat += (1.0 - j / (lag + 1.0)) * (cross + cross.T)
    cov = bread @ meat @ bread * n / dof
    return float(beta[0]), float(beta[1]), float(np.sqrt(cov[0, 0])), float(sigma_ols)

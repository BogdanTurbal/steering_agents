"""Homogenization metrics (paper Defs. 3.2-3.4) from a score tensor [runs, T, agents] (NaN = missing)."""
import numpy as np


def conformity_curve(S):
    """Inter-agent variance per round, averaged over runs. S: [R, T, A]."""
    return np.nanmean(np.nanvar(S, axis=2, ddof=0), axis=0)


def polarization_curve(S, rmin, rmax):
    """Distance of the panel mean to the nearest scale endpoint, per round (smaller = more polarized)."""
    m = np.nanmean(S, axis=2)
    return np.nanmean(np.minimum(np.abs(m - rmin), np.abs(m - rmax)), axis=0)


def inertia_curve(S):
    """|change of the panel mean| per round transition, averaged over runs (smaller = more inertia)."""
    m = np.nanmean(S, axis=2)
    return np.nanmean(np.abs(np.diff(m, axis=1)), axis=0)


def definition_rates(S, rmin, rmax):
    """Fraction of (run, transition) pairs satisfying Defs 3.2-3.4 literally (strict inequalities)."""
    var = np.nanvar(S, axis=2)
    m = np.nanmean(S, axis=2)
    dist = np.minimum(np.abs(m - rmin), np.abs(m - rmax))
    step = np.abs(np.diff(m, axis=1))
    return {"conformity": float(np.nanmean(var[:, 1:] < var[:, :-1])),
            "polarization": float(np.nanmean(dist[:, 1:] < dist[:, :-1])),
            "inertia": float(np.nanmean(step[:, 1:] < step[:, :-1]))}

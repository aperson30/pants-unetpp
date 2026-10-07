"""CPU-only exact-score toy screen; no medical/model novelty claim.

VE marginal p_sigma is a two-Gaussian mixture. Start from its EXACT finite-noise
quantiles, not an approximate normal prior. In one dimension the probability
flow preserves quantile rank, supplying exact endpoints without a neural judge.
All solver calls (including intermediate stages) count toward the NFE budget.
This is a fixed heuristic baseline screen, NOT AYS/INDIS reproduction or tuning.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtr, logsumexp


def cdf(x, sigma, weight, separation):
    scale = np.sqrt(0.35**2 + sigma**2)
    return (1-weight)*ndtr(x/scale) + weight*ndtr((x-separation)/scale)


def quantiles(u, sigma, weight, separation):
    scale = np.sqrt(0.35**2 + sigma**2)
    lo = np.full_like(u, -12*scale)
    hi = np.full_like(u, separation+12*scale)
    for _ in range(85):
        mid = (lo+hi)/2
        below = cdf(mid, sigma, weight, separation) < u
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    out = (lo+hi)/2
    assert np.max(np.abs(cdf(out, sigma, weight, separation)-u)) < 1e-12
    return out


def velocity(x, sigma, weight, separation):
    variance = 0.35**2 + sigma**2
    log_weights = np.stack([
        np.log1p(-weight)-x*x/(2*variance),
        np.log(weight)-(x-separation)**2/(2*variance)])
    responsibility = np.exp(log_weights[1]-logsumexp(log_weights, axis=0))
    score = (responsibility*separation-x)/variance
    return -sigma*score


def integrate(x, times, method, weight, separation):
    x = x.copy()
    calls = 0
    for start, end in zip(times[:-1], times[1:]):
        h = end-start
        k1 = velocity(x, start, weight, separation)
        calls += 1
        if method == 'euler':
            x += h*k1
        elif method == 'heun':
            k2 = velocity(x+h*k1, end, weight, separation)
            x += h*(k1+k2)/2
            calls += 1
        elif method == 'rk4':
            k2 = velocity(x+h*k1/2, start+h/2, weight, separation)
            k3 = velocity(x+h*k2/2, start+h/2, weight, separation)
            k4 = velocity(x+h*k3, end, weight, separation)
            x += h*(k1+2*k2+2*k3+k4)/6
            calls += 3
        else:
            raise ValueError(method)
    assert np.isfinite(x).all()
    return x, calls


def schedule(kind, steps):
    if kind == 'linear':
        return np.linspace(10., .01, steps+1)
    if kind == 'geometric':
        return np.geomspace(10., .01, steps+1)
    if kind == 'rho7':
        return np.linspace(10.**(1/7), .01**(1/7), steps+1)**7
    raise ValueError(kind)


def contracts():
    # Score agrees with a finite-difference log density, including mixed modes.
    x = np.linspace(-2, 6, 39)
    sig, w, sep, eps = .8, .1, 4., 1e-5
    var = .35**2+sig**2
    def logpdf(z):
        return logsumexp(np.stack([np.log1p(-w)-z*z/(2*var),
                                  np.log(w)-(z-sep)**2/(2*var)]), axis=0)
    np.testing.assert_allclose(-velocity(x, sig, w, sep)/sig,
        (logpdf(x+eps)-logpdf(x-eps))/(2*eps), atol=1e-9, rtol=1e-8)
    # Identical components reduce to a single Gaussian with known exact transport.
    initial = np.linspace(-20, 20, 31)
    exact = initial*np.sqrt((.35**2+.01**2)/(.35**2+10.**2))
    output, calls = integrate(initial, schedule('rho7', 256), 'rk4', .1, 0.)
    assert calls == 1024
    np.testing.assert_allclose(output, exact, atol=1e-7, rtol=1e-7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), 'Never overwrite a prior screen'
    start = time.perf_counter()
    contracts()
    rows = []
    # Fixed before outcomes: every weight/separation/budget/solver/grid retained.
    # Midpoint quadrature, not independent patient samples or stochastic seeds.
    u = (np.arange(4096)+.5)/4096
    for weight in (.01, .1, .5):
        for separation in (2., 4.):
            initial = quantiles(u, 10., weight, separation)
            exact = quantiles(u, .01, weight, separation)
            positive = exact > separation/2
            exact_mass = 1-float(cdf(separation/2, .01, weight, separation))
            for nfe in (16, 32, 64, 128):
                for method, stages in (('euler', 1), ('heun', 2), ('rk4', 4)):
                    for grid in ('linear', 'geometric', 'rho7'):
                        output, calls = integrate(initial, schedule(grid, nfe//stages),
                                                  method, weight, separation)
                        assert calls == nfe
                        predicted = output > separation/2
                        rows.append(dict(weight=weight, separation=separation,
                            nfe=nfe, method=method, grid=grid,
                            global_mse=float(np.mean((output-exact)**2)),
                            exact_positive_region_mse=float(np.mean((output[positive]-exact[positive])**2)),
                            transported_positive_miss_fraction=float(np.mean(~predicted[positive])),
                            transported_negative_cross_fraction=float(np.mean(predicted[~positive])),
                            predicted_positive_mass=float(predicted.mean()),
                            exact_positive_mass=exact_mass,
                            absolute_mass_error=float(abs(predicted.mean()-exact_mass))))
    report = dict(schema='exact-score-rare-mode-solver-screen-v1',
        scope='1D VE analytic toy only; no spatial lesions, learned scores, or clinical endpoint',
        quadrature_points=4096, contracts_passed=True, rows=rows,
        cpu_wall_seconds=time.perf_counter()-start, gpu_hours=0,
        caveats=['Grid integration error is separated from score/prior error.',
                 'Quadrature mass resolution is 1/4096; not a confidence bound.',
                 'No schedule optimization; no claim of beating AYS, INDIS or DPM-Solver.',
                 'Positive region is a fixed geometric region, not latent mixture identity.'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(rows=len(rows), seconds=report['cpu_wall_seconds'], contracts=True)))


if __name__ == '__main__':
    main()

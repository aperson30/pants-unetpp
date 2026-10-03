"""Exact CPU feasibility checks, NOT trained-model evidence or a new theorem."""
import json
import math
import unittest


def moments(full, shallow, p):
    # Exact Bernoulli enumeration: no Monte Carlo seed or sampling error.
    values = [(shallow, 1-p), (shallow+(full-shallow)/p, p)]
    mean = sum(x*w for x, w in values)
    var = sum(w*(x-mean)**2 for x, w in values)
    return mean, var


def efficiency(a, b, r, p):
    # Vfull normalized to1; r=E||gf-gs||^2 / trace(Cov(gf)).
    # a=always-paid shallow cost; b=incremental full correction cost.
    # Both include backward; any extra correction overhead belongs in b.
    return (a+p*b) * (1+(1/p-1)*r)


def optimum(a, b, r):
    if r == 0:
        return 0., a
    if r >= 1:
        return 1., a+b
    p = min(1., math.sqrt(a*r/(b*(1-r))))
    return p, efficiency(a, b, r, p)


class Contracts(unittest.TestCase):
    def test_mean_and_added_variance(self):
        for p in [.05, .25, .5, 1.]:
            for f, s in [(2., 0.), (-1., .4), (3., 3.)]:
                m, v = moments(f, s, p)
                self.assertAlmostEqual(m, f)
                self.assertAlmostEqual(v, (1/p-1)*(f-s)**2)

    def test_clip_is_not_preserved(self):
        # full2, shallow0, p=.25 -> estimator0 or8; cap1.
        self.assertEqual(.75*0+.25*1, .25)
        self.assertNotEqual(.25, min(2., 1.))

    def test_analytic_optimum(self):
        for a in [.1, .25, .5]:
            for r in [.01, .1, .3, .7, 1., 2.]:
                p, value = optimum(a, 1-a, r)
                grid = min(efficiency(a, 1-a, r, k/10000)
                           for k in range(1, 10001))
                self.assertLessEqual(value, grid+1e-12)
                self.assertLessEqual(value, 1.+1e-12)


def report():
    return dict(
        scope='Analytic feasibility only; no empirical gradient/runtimes',
        gpu_hours=0,
        clipping_example=dict(full_clipped=1., expected_estimator_clipped=.25),
        sweeps=[dict(shallow_cost=a, residual_second_moment_ratio=r,
                     optimal_full_probability=optimum(a, 1-a, r)[0],
                     cost_times_variance_relative=optimum(a, 1-a, r)[1])
                for a in [.1, .25, .5] for r in [.01, .1, .3, .7, 1., 2.]],
        caveats=['Cost-times-variance is not SGD convergence or sample quality',
                 'Unbiased raw gradients do not preserve clipped/Adam updates',
                 'Residual second moment is not residual centered variance',
                 'Deep-only parameters receive no surrogate gradient'])


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    result = unittest.TextTestRunner().run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(json.dumps(report(), indent=2))

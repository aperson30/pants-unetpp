"""Exact statistical/metric checks. Synthetic only, no patient conclusions."""
import json
import math
import unittest


def zero_failure_upper(n, delta=.05):
    if n < 1 or not 0 < delta < 1:
        raise ValueError('Need positive sample count and confidence parameter')
    return -math.expm1(math.log(delta)/n)


def required_zero_failure_cases(tolerance, delta=.05):
    if not 0 < tolerance < 1:
        raise ValueError('Tolerance must be between0 and1')
    return math.ceil(math.log(delta)/math.log1p(-tolerance))


class Checks(unittest.TestCase):
    def test_exact_minimum(self):
        for target in [.1, .05, .01]:
            n = required_zero_failure_cases(target)
            self.assertLessEqual(zero_failure_upper(n), target)
            self.assertGreater(zero_failure_upper(n-1), target)

    def test_marginal_is_not_positive_case_risk(self):
        prevalence, conditional_miss = .1, .4
        marginal = prevalence*conditional_miss
        self.assertLessEqual(marginal, .05)
        self.assertGreater(conditional_miss, .05)

    def test_pixel_recall_does_not_control_each_lesion(self):
        large, tiny = 10000, 10
        # Perfect large lesion, complete small lesion miss.
        voxel_recall = large/(large+tiny)
        self.assertGreater(voxel_recall, .999)
        self.assertEqual(1/2, .5)  # Whole-lesion recall is only50%.


if __name__ == '__main__':
    result = unittest.TextTestRunner().run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(json.dumps(dict(
        scope='Known binomial bounds and synthetic metric counterexamples, not new theory',
        zero_failure_positive_cases={str(t): required_zero_failure_cases(t)
                                     for t in [.1, .05, .01]},
        n10_upper=zero_failure_upper(10),
        caveats=['Fixed prespecified policy only; adaptive threshold search needs correction',
                 'Independent cases, not correlated lesions/patches within a patient',
                 'Zero failures is optimistic; nonzero counts require different bounds',
                 'This is high-probability binomial validation, not CRC expected-risk calibration',
                 'Newly introduced miss relative to full model is not absolute clinical sensitivity']),
        indent=2))

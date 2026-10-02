import unittest
from select_judge_replication import select


def case(study, size, fold=4, reference='histopathology'):
    return dict(study=study, fold=fold, reference=reference,
                archive=dict(compressed=size, encrypted=False, compression=0))


class SelectionTests(unittest.TestCase):
    def test_fixed_eligibility_order_and_pilot_exclusion(self):
        audit = dict(candidates=[case('b', 40), case('100226_00001', 1),
                                case('a', 30), case('other_fold', 2, fold=1),
                                case('cytology', 3, reference='cytology'),
                                case('large', 60_000_001)])
        self.assertEqual([c['study'] for c in select(audit)], ['a', 'b'])
        audit['candidates'].reverse()
        self.assertEqual([c['study'] for c in select(audit)], ['a', 'b'])

    def test_insufficient_cases_refused(self):
        with self.assertRaises(ValueError):
            select(dict(candidates=[case('a', 20)]))


if __name__ == '__main__':
    unittest.main()

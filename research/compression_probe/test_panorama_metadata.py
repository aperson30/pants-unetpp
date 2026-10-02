import unittest
from audit_panorama_metadata import normalize_folds


class FoldContracts(unittest.TestCase):
    def test_order_does_not_change_membership(self):
        a = {f'Fold {k} validation': [str(2*k), str(2*k+1)] for k in range(5)}
        b = {key: list(reversed(value)) for key, value in a.items()}
        self.assertEqual(normalize_folds(a), normalize_folds(b))

    def test_duplicate_and_cross_fold_overlap_rejected(self):
        a = {f'Fold {k} validation': [str(k)] for k in range(5)}
        a['Fold 0 validation'].append('0')
        with self.assertRaises(ValueError):
            normalize_folds(a)
        a['Fold 0 validation'] = ['1']
        with self.assertRaises(ValueError):
            normalize_folds(a)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
import unittest

from placement_experiment import MAX, R1, solve, validate


def request(root, first, last, size=8192):
    return {'class': 'Ipu21Standard', 'region1_stride': None,
            'lifetime': {'first': first, 'last': last, 'seen': True},
            'bytes': size, 'alignment': 32, 'assignments': [[root, 0]], 'conflicts': []}


class PlacementExperimentTests(unittest.TestCase):
    def setUp(self):
        self.data = {'ranges': [[R1, R1 + 32768]], 'interleaved_offset': 0,
                     'requests': [request(4, 1, 3), request(9, 4, 6)]}
        self.reused = {0: (R1, R1 + 8192), 1: (R1, R1 + 8192)}

    def test_reuse_requires_disjoint_lifetimes(self):
        validate(self.data, self.reused)
        self.data['requests'][1]['lifetime']['first'] = 3
        with self.assertRaises(AssertionError):
            validate(self.data, self.reused)

    def test_bank_conflicts_are_stronger_than_address_disjointness(self):
        self.data['requests'][0]['conflicts'] = [9]
        with self.assertRaises(AssertionError):
            validate(self.data, {0: (R1, R1 + 8192), 1: (R1 + 8192, R1 + 16384)})

    def test_stride_and_alignment_are_checked(self):
        self.data['requests'][0]['assignments'] = [[4, 0], [5, 4096]]
        self.data['requests'][0]['region1_stride'] = 32768
        with self.assertRaises(AssertionError):
            validate(self.data, self.reused)

    def test_bound_is_distinct_from_exhausted_search(self):
        self.data['requests'] = [request(4, 0, MAX, 24576), request(9, 0, MAX, 24576)]
        self.assertEqual(solve(self.data)['status'], 'infeasible_bound')
        self.data['requests'] = [request(4, 0, MAX)]
        self.assertEqual(solve(self.data, budget=0)['status'], 'unknown')
        self.assertEqual(solve(self.data)['status'], 'fit')


if __name__ == '__main__':
    unittest.main()

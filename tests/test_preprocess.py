#!/usr/bin/env python3
"""
Basic test for preprocess script.
"""

import unittest
import pandas as pd
import numpy as np
from src.preprocess import load_and_normalize_votes, compute_tallies

class TestPreprocess(unittest.TestCase):
    def test_load_votes(self):
        # Mock data
        df = pd.DataFrame({
            'ImageID': [1, 1, 2],
            'MobilityAid': ['Cane', 'Cane', 'Cane'],
            'Selection': ['yes', 'no', 'unsure']
        })
        df.to_csv('/tmp/test_votes.csv', index=False)
        result = load_and_normalize_votes('/tmp/test_votes.csv')
        self.assertEqual(len(result), 3)

    def test_compute_tallies(self):
        df_votes = pd.DataFrame({
            'ImageID': [1, 1, 1],
            'MobilityAid': ['Walking cane', 'Walking cane', 'Walking cane'],
            'Selection': ['yes', 'no', 'yes']
        })
        tallies = compute_tallies(df_votes)
        self.assertIn('Walking cane', tallies['MobilityAid'].values)

if __name__ == '__main__':
    unittest.main()
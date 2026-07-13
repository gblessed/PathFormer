import numpy as np
import unittest

from pathformer_pilot_geometry_channel_estimation import (
    reconstruct_from_sparse_pilots,
    select_pilot_indices,
)


class PilotGeometryEstimatorTest(unittest.TestCase):
    def test_reconstruct_from_sparse_pilots_recovers_known_complex_coefficients(self):
        rng = np.random.default_rng(7)
        basis = rng.normal(size=(32, 4)) + 1j * rng.normal(size=(32, 4))
        coeffs = np.array([1.0 + 0.5j, -0.3 + 0.2j, 0.7 - 0.1j, -0.2j], dtype=np.complex64)
        channel = basis @ coeffs
        pilot_idx = np.arange(32)

        reconstructed, estimated_coeffs = reconstruct_from_sparse_pilots(
            basis,
            channel,
            pilot_idx,
            ridge_lambda=1e-8,
        )

        np.testing.assert_allclose(estimated_coeffs, coeffs, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(reconstructed, channel, rtol=1e-4, atol=1e-4)

    def test_select_pilot_indices_is_deterministic_and_unique(self):
        idx_a = select_pilot_indices(total_entries=100, pilot_count=12, seed=123)
        idx_b = select_pilot_indices(total_entries=100, pilot_count=12, seed=123)

        self.assertEqual(idx_a.shape, (12,))
        self.assertTrue(np.array_equal(idx_a, idx_b))
        self.assertEqual(len(np.unique(idx_a)), 12)


if __name__ == "__main__":
    unittest.main()

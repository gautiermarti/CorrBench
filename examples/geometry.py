"""CPU geometry round trip; no market data or fitted model required."""

import numpy as np

from corrbench.geometry import corr_to_gft, gft_to_corr, is_correlation_matrix

matrix = np.array([[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]])
restored = gft_to_corr(corr_to_gft(matrix))
np.testing.assert_allclose(restored, matrix, atol=1e-7)
assert is_correlation_matrix(restored)
print("GFT round trip valid; maximum error:", np.abs(restored - matrix).max())

import numpy as np

from corrbench.ordering import ward_order, ward_permutation


def test_ward_order_is_a_deterministic_permutation_similarity() -> None:
    correlation = np.array(
        [
            [1.0, 0.8, 0.1, 0.2],
            [0.8, 1.0, 0.2, 0.1],
            [0.1, 0.2, 1.0, 0.7],
            [0.2, 0.1, 0.7, 1.0],
        ],
        dtype=np.float32,
    )
    ordered, permutation = ward_order(correlation)
    assert sorted(permutation.tolist()) == list(range(4))
    np.testing.assert_array_equal(permutation, ward_permutation(correlation))
    np.testing.assert_allclose(ordered, correlation[np.ix_(permutation, permutation)])
    np.testing.assert_allclose(
        np.linalg.eigvalsh(ordered), np.linalg.eigvalsh(correlation), atol=1e-6
    )


def test_ward_order_rejects_non_square_input() -> None:
    with np.testing.assert_raises(ValueError):
        ward_permutation(np.zeros((3, 4), dtype=np.float32))

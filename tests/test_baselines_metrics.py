import numpy as np

from corrbench.baselines import (
    ConditionalGaussian,
    ConditionalLinearPCAGaussian,
    empirical_bootstrap,
    nearest_covariate_bootstrap,
)
from corrbench.metrics import (
    classifier_two_sample_auc,
    manifold_precision_recall,
    nearest_train_diagnostic,
    orbit_sliced_wasserstein,
    portfolio_functionals,
    structural_features,
    validity_summary,
)


def test_conditional_gaussian_shapes_and_class_means() -> None:
    rng = np.random.default_rng(2)
    labels = np.repeat([0, 1], 100)
    values = rng.normal(size=(200, 6)) + labels[:, None] * 3
    model = ConditionalGaussian(coordinatewise=False)
    model.fit(values, labels)
    samples = model.sample(np.repeat([0, 1], 500), np.random.default_rng(3))
    assert samples.shape == (1000, 6)
    assert samples[500:].mean() - samples[:500].mean() > 2


def test_bootstrap_only_draws_from_matching_class() -> None:
    values = np.arange(24).reshape(6, 4).astype(float)
    labels = np.array([0, 0, 0, 1, 1, 1])
    target = np.array([0, 1] * 10)
    sampled = empirical_bootstrap(values, labels, target, np.random.default_rng(1))
    assert np.all(sampled[target == 0, 0] < 12)
    assert np.all(sampled[target == 1, 0] >= 12)


def test_covariate_conditioned_baselines_follow_target() -> None:
    rng = np.random.default_rng(12)
    labels = np.repeat([0, 1], 120)
    covariate = np.tile(np.linspace(-2, 2, 120), 2)
    values = covariate[:, None] * np.array([[2.0, -1.0]])
    values += labels[:, None] * 10 + rng.normal(scale=0.05, size=(240, 2))
    targets = np.array([0, 0, 1, 1])
    target_covariate = np.array([-1.8, 1.8, -1.8, 1.8])
    replay = nearest_covariate_bootstrap(
        values,
        labels,
        covariate,
        targets,
        target_covariate,
        rng,
        neighbors=5,
    )
    assert replay[1, 0] > replay[0, 0]
    assert replay[3, 0] > replay[2, 0]

    model = ConditionalLinearPCAGaussian(components=1, random_state=3)
    model.fit(values, labels, covariate)
    sampled = model.sample(targets, target_covariate, rng)
    assert sampled[1, 0] > sampled[0, 0] + 5
    assert sampled[3, 0] > sampled[2, 0] + 5


def test_structural_features_are_permutation_invariant() -> None:
    corr = np.array(
        [
            [1.0, 0.2, -0.1],
            [0.2, 1.0, 0.4],
            [-0.1, 0.4, 1.0],
        ]
    )[None]
    permutation = np.array([2, 0, 1])
    features, names = structural_features(corr)
    permuted, _ = structural_features(corr[:, permutation][:, :, permutation])
    np.testing.assert_allclose(features, permuted)
    assert features.shape[1] == len(names)
    assert validity_summary(corr)["valid_fraction"] == 1.0


def test_portfolio_functionals_are_permutation_invariant() -> None:
    rng = np.random.default_rng(4)
    loadings = rng.normal(size=(8, 5, 3))
    covariance = loadings @ np.swapaxes(loadings, -1, -2) + np.eye(5)
    scale = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
    matrices = covariance / scale[:, :, None] / scale[:, None, :]
    permutation = np.array([3, 0, 4, 1, 2])
    permuted = matrices[:, permutation][:, :, permutation]
    original, names = portfolio_functionals(matrices)
    transformed, _ = portfolio_functionals(permuted)
    assert len(names) == 4
    np.testing.assert_allclose(original, transformed, rtol=1e-10, atol=1e-10)


def test_classifier_and_nearest_train_diagnostics() -> None:
    rng = np.random.default_rng(5)
    real = rng.normal(size=(100, 4))
    shifted = rng.normal(loc=3.0, size=(100, 4))
    result = classifier_two_sample_auc(real, shifted, folds=4, seed=5)
    assert result["auc"] > 0.95
    nearest = nearest_train_diagnostic(real, shifted, real[:20], seed=5)
    assert nearest["generated_exact_match_fraction"] == 1.0


def test_manifold_precision_recall_detects_shift() -> None:
    rng = np.random.default_rng(8)
    real = rng.normal(size=(500, 3))
    matched = manifold_precision_recall(real, real.copy(), seed=4)
    shifted = manifold_precision_recall(real, real + 20, seed=4)
    assert matched == {"precision": 1.0, "recall": 1.0}
    assert shifted == {"precision": 0.0, "recall": 0.0}


def test_orbit_distance_ignores_a_permutation() -> None:
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(40, 5, 5))
    matrix = matrix + matrix.transpose(0, 2, 1)
    np.fill_diagonal(matrix[0], 0)
    indices = np.tril_indices(5, k=-1)
    vectors = matrix[:, indices[0], indices[1]]
    # The distributions are identical; independent orbit randomization should
    # leave only finite-sample projection noise rather than positional bias.
    distance = orbit_sliced_wasserstein(
        vectors, vectors.copy(), projections=16, seed=2, device="cpu"
    )
    assert distance < 1.0

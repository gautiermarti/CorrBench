"""Transparent statistical baselines in generalized-Fisher coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA

FloatArray = NDArray[np.floating]


@dataclass
class ConditionalGaussian:
    """Class-conditional Gaussian with iid or coordinatewise scale."""

    coordinatewise: bool = False
    variance_floor: float = 1e-6

    def fit(self, values: FloatArray, labels: NDArray[np.integer]) -> None:
        self.dimension_ = values.shape[1]
        self.classes_ = np.unique(labels)
        self.means_: dict[int, FloatArray | float] = {}
        self.scales_: dict[int, FloatArray | float] = {}
        for label in self.classes_:
            selected = values[labels == label].astype(np.float64)
            if self.coordinatewise:
                mean = selected.mean(axis=0)
                scale = selected.std(axis=0, ddof=1)
            else:
                mean = float(selected.mean())
                scale = float(selected.std(ddof=1))
            self.means_[int(label)] = mean
            self.scales_[int(label)] = np.maximum(scale, self.variance_floor)

    def sample(
        self, labels: NDArray[np.integer], rng: np.random.Generator
    ) -> FloatArray:
        if not hasattr(self, "classes_"):
            raise RuntimeError("Fit the baseline before sampling.")
        dimension = self.dimension_
        output = np.empty((labels.size, dimension), dtype=np.float32)
        for label in np.unique(labels):
            mask = labels == label
            output[mask] = rng.normal(
                self.means_[int(label)],
                self.scales_[int(label)],
                size=(int(mask.sum()), dimension),
            )
        return output


@dataclass
class ConditionalPCAGaussian:
    """Low-rank Gaussian plus isotropic residual, fitted independently by class."""

    components: int = 64
    variance_floor: float = 1e-6
    random_state: int = 0

    def fit(self, values: FloatArray, labels: NDArray[np.integer]) -> None:
        self.dimension_ = values.shape[1]
        self.models_: dict[int, PCA] = {}
        self.residual_scales_: dict[int, float] = {}
        for label in np.unique(labels):
            selected = values[labels == label]
            model = PCA(
                n_components=min(self.components, selected.shape[0] - 1),
                svd_solver="randomized",
                random_state=self.random_state + int(label),
            ).fit(selected)
            explained = float(model.explained_variance_.sum())
            total = float(selected.var(axis=0, ddof=1).sum())
            residual_variance = max(
                (total - explained) / max(self.dimension_ - model.n_components_, 1),
                self.variance_floor**2,
            )
            self.models_[int(label)] = model
            self.residual_scales_[int(label)] = float(np.sqrt(residual_variance))

    def sample(
        self, labels: NDArray[np.integer], rng: np.random.Generator
    ) -> FloatArray:
        output = np.empty((labels.size, self.dimension_), dtype=np.float32)
        for label in np.unique(labels):
            mask = labels == label
            model = self.models_[int(label)]
            count = int(mask.sum())
            latent = rng.normal(size=(count, model.n_components_))
            latent *= np.sqrt(model.explained_variance_)
            reconstructed = latent @ model.components_ + model.mean_
            residual = rng.normal(
                scale=self.residual_scales_[int(label)],
                size=reconstructed.shape,
            )
            output[mask] = reconstructed + residual
        return output


@dataclass
class ConditionalLinearPCAGaussian:
    """Classwise linear conditional mean plus a low-rank Gaussian residual."""

    components: int = 64
    variance_floor: float = 1e-6
    random_state: int = 0

    def fit(
        self,
        values: FloatArray,
        labels: NDArray[np.integer],
        covariate: FloatArray,
    ) -> None:
        self.dimension_ = values.shape[1]
        self.models_: dict[int, PCA] = {}
        self.intercepts_: dict[int, FloatArray] = {}
        self.slopes_: dict[int, FloatArray] = {}
        self.residual_scales_: dict[int, float] = {}
        for label in np.unique(labels):
            mask = labels == label
            selected = values[mask].astype(np.float64)
            x = np.asarray(covariate[mask], dtype=np.float64)
            x_mean = float(x.mean())
            centered_x = x - x_mean
            denominator = max(float(centered_x @ centered_x), self.variance_floor)
            slope = centered_x @ (selected - selected.mean(axis=0)) / denominator
            intercept = selected.mean(axis=0) - slope * x_mean
            residual = selected - intercept - x[:, None] * slope
            model = PCA(
                n_components=min(self.components, residual.shape[0] - 1),
                svd_solver="randomized",
                random_state=self.random_state + int(label),
            ).fit(residual)
            explained = float(model.explained_variance_.sum())
            total = float(residual.var(axis=0, ddof=1).sum())
            residual_variance = max(
                (total - explained) / max(self.dimension_ - model.n_components_, 1),
                self.variance_floor**2,
            )
            key = int(label)
            self.models_[key] = model
            self.intercepts_[key] = intercept
            self.slopes_[key] = slope
            self.residual_scales_[key] = float(np.sqrt(residual_variance))

    def sample(
        self,
        labels: NDArray[np.integer],
        covariate: FloatArray,
        rng: np.random.Generator,
    ) -> FloatArray:
        output = np.empty((labels.size, self.dimension_), dtype=np.float32)
        for label in np.unique(labels):
            mask = labels == label
            key = int(label)
            model = self.models_[key]
            count = int(mask.sum())
            latent = rng.normal(size=(count, model.n_components_))
            latent *= np.sqrt(model.explained_variance_)
            residual = latent @ model.components_ + model.mean_
            residual += rng.normal(
                scale=self.residual_scales_[key], size=residual.shape
            )
            conditional_mean = (
                self.intercepts_[key]
                + np.asarray(covariate[mask])[:, None] * self.slopes_[key]
            )
            output[mask] = conditional_mean + residual
        return output


def empirical_bootstrap(
    values: FloatArray,
    train_labels: NDArray[np.integer],
    target_labels: NDArray[np.integer],
    rng: np.random.Generator,
) -> FloatArray:
    """Sample training observations with replacement within each class."""
    output = np.empty((target_labels.size, values.shape[1]), dtype=np.float32)
    for label in np.unique(target_labels):
        target_mask = target_labels == label
        candidates = np.flatnonzero(train_labels == label)
        selected = rng.choice(candidates, size=int(target_mask.sum()), replace=True)
        output[target_mask] = values[selected]
    return output


def nearest_covariate_bootstrap(
    values: FloatArray,
    train_labels: NDArray[np.integer],
    train_covariate: FloatArray,
    target_labels: NDArray[np.integer],
    target_covariate: FloatArray,
    rng: np.random.Generator,
    *,
    neighbors: int = 100,
) -> FloatArray:
    """Replay a random training observation among nearest conditional values."""
    output = np.empty((target_labels.size, values.shape[1]), dtype=np.float32)
    for label in np.unique(target_labels):
        candidates = np.flatnonzero(train_labels == label)
        order = candidates[np.argsort(train_covariate[candidates])]
        sorted_covariate = train_covariate[order]
        for target in np.flatnonzero(target_labels == label):
            position = int(np.searchsorted(sorted_covariate, target_covariate[target]))
            half = min(neighbors, order.size) // 2
            start = max(
                0,
                min(position - half, order.size - min(neighbors, order.size)),
            )
            local = order[start : start + min(neighbors, order.size)]
            output[target] = values[int(rng.choice(local))]
    return output

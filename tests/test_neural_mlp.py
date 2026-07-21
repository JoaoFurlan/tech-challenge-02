"""Testes de fumaça para o MLP (`NeuralMLPClassifier`/`NeuralMLPModel`)."""

import numpy as np
import pandas as pd

from recsys_ecommerce.models.neural_mlp import NeuralMLPClassifier, NeuralMLPModel


def _toy_data(
    n: int = 200, n_features: int = 4, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)]
    )
    y = pd.Series((X["f0"] + X["f1"] > 0).astype(int))
    return X, y


def test_fit_predict_proba_shapes() -> None:
    X, y = _toy_data()
    clf = NeuralMLPClassifier(hidden_dims=(8,), max_epochs=3, patience=3)
    clf.fit(X, y)
    probabilities = clf.predict_proba(X)
    assert probabilities.shape == (len(X), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)


def test_pos_weight_computed_when_weighted_loss() -> None:
    X, y = _toy_data()
    clf = NeuralMLPClassifier(
        hidden_dims=(8,), max_epochs=2, patience=2, weighted_loss=True
    )
    clf.fit(X, y)
    assert clf.pos_weight_ > 0


def test_pos_weight_is_one_when_not_weighted() -> None:
    X, y = _toy_data()
    clf = NeuralMLPClassifier(
        hidden_dims=(8,), max_epochs=2, patience=2, weighted_loss=False
    )
    clf.fit(X, y)
    assert clf.pos_weight_ == 1.0


def test_neural_mlp_model_wraps_classifier_via_tabular_classifier() -> None:
    X, y = _toy_data()
    model = NeuralMLPModel(hidden_dims=(8,), max_epochs=2, patience=2)
    model.fit(X, y)
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (len(X), 2)
    assert "hidden_dims" in model.get_params()


def test_loss_history_is_recorded() -> None:
    X, y = _toy_data()
    clf = NeuralMLPClassifier(hidden_dims=(8,), max_epochs=4, patience=4)
    clf.fit(X, y)
    assert len(clf.loss_history_["train"]) == clf.n_epochs_trained_
    assert len(clf.loss_history_["early_stop"]) == clf.n_epochs_trained_

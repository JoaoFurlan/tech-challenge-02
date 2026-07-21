"""Testes de fumaça para os padrões Factory/Strategy/Template Method."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from recsys_ecommerce.models.factory import ModelFactory
from recsys_ecommerce.models.tabular_classifier import TabularClassifierModel
from recsys_ecommerce.preprocessing.base import Preprocessor


class _NoOpPreprocessor(Preprocessor[pd.DataFrame]):
    """Strategy trivial só para exercitar a interface `Preprocessor`."""

    def fit(self, data: pd.DataFrame) -> "_NoOpPreprocessor":
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return data


def test_preprocessor_fit_transform_roundtrip() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = _NoOpPreprocessor().fit_transform(df)
    pd.testing.assert_frame_equal(result, df)


class _TestLogisticRegressionModel(TabularClassifierModel):
    """Subclasse mínima só para exercitar `ModelFactory` nos testes."""

    def __init__(self) -> None:
        super().__init__(LogisticRegression())


def test_model_factory_register_and_create() -> None:
    ModelFactory.register("_test_logreg", _TestLogisticRegressionModel)
    model = ModelFactory.create("_test_logreg")
    assert isinstance(model, TabularClassifierModel)


def test_model_factory_rejects_duplicate_registration() -> None:
    ModelFactory.register("_test_dup", _TestLogisticRegressionModel)
    with pytest.raises(ValueError, match="(?i)já existe"):
        ModelFactory.register("_test_dup", _TestLogisticRegressionModel)


def test_model_factory_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="não registrado"):
        ModelFactory.create("_nome_que_nao_existe")


def test_tabular_classifier_fit_predict_proba() -> None:
    X = pd.DataFrame({"f1": [0.0, 1.0, 0.0, 1.0], "f2": [1.0, 1.0, 0.0, 0.0]})
    y = pd.Series([0, 1, 0, 1])
    model = TabularClassifierModel(LogisticRegression())
    model.fit(X, y)
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (4, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert model.get_params()["C"] == pytest.approx(1.0)

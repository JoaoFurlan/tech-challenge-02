"""Classe-base para modelos que envolvem um estimador sklearn-compatible.

Qualquer estimador com `fit(X, y)`/`predict_proba(X)` pode ser injetado --
centraliza o Template Method (`_fit_estimator`, sobrescrevível por
subclasses que precisam de argumentos extras no treino, ex. `eval_set` para
early stopping) e delega tudo o mais direto ao estimador injetado.
Compartilhada por `LogisticRegressionBaseline`, `DecisionTreeModel`,
`XGBoostModel`, `LightGBMModel` — e também pelo MLP: como
`NeuralMLPClassifier` (`recsys_ecommerce.models.neural_mlp`) já é
sklearn-like, ele encaixa aqui sem precisar de uma subclasse própria de
`RecommenderModel`.
"""

from typing import Any, Self

import numpy as np
import pandas as pd

from recsys_ecommerce.models.base import RecommenderModel


class TabularClassifierModel(RecommenderModel):
    """Recomendador genérico: um classificador sklearn-compatible sobre features tabulares.

    Subclasses só precisam construir o estimador certo e chamar
    `super().__init__(estimator=...)` — todo o resto (treino, scoring) é
    compartilhado aqui.
    """

    def __init__(self, estimator: Any) -> None:  # noqa: ANN401
        """Inicializa o modelo.

        Args:
            estimator: Estimador sklearn-compatible (com `fit(X, y)` e
                `predict_proba(X)`), já configurado pela subclasse.
        """
        self._clf = estimator

    @property
    def underlying_estimator(self) -> Any:  # noqa: ANN401
        """Estimador treinado, para logging via `mlflow.sklearn.log_model`."""
        return self._clf

    def get_params(self) -> dict[str, Any]:  # noqa: ANN401
        """Repassa os hiperparâmetros do estimador interno, para logging no MLflow."""
        get_params = getattr(self._clf, "get_params", None)
        return dict(get_params()) if get_params is not None else {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """Treina o classificador. Ver `RecommenderModel.fit`."""
        self._fit_estimator(X, y)
        return self

    def _fit_estimator(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Treina `self._clf`.

        Sobrescrita por subclasses que precisam passar argumentos extras
        (ex. `eval_set` para early stopping em xgboost/lightgbm).
        """
        self._clf.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna P(interação) por linha. Ver `RecommenderModel.predict_proba`."""
        result: np.ndarray = self._clf.predict_proba(X)
        return result

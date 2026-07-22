"""XGBoost sobre as features tabulares (`fe_v4`)."""

from typing import Self

import mlflow
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from recsys_ecommerce.models.tabular_classifier import TabularClassifierModel


class XGBoostModel(TabularClassifierModel):
    """Gradient boosting (XGBoost), injetado em `TabularClassifierModel`."""

    def __init__(self, **kwargs: object) -> None:
        """Inicializa o modelo.

        Args:
            **kwargs: Repassados a `xgboost.XGBClassifier` (ex.: `max_depth`,
                `learning_rate`, `n_estimators`, `subsample`,
                `colsample_bytree`, `reg_alpha`, `reg_lambda`, `random_state`).
        """
        kwargs.setdefault("eval_metric", "logloss")
        super().__init__(XGBClassifier(**kwargs))
        self._eval_data: tuple[pd.DataFrame, pd.Series] | None = None

    def set_periodic_eval(
        self,
        val_feat: pd.DataFrame,
        test_feat: pd.DataFrame,
        feature_columns: list[str],
        all_items: np.ndarray,
    ) -> Self:
        """Liga o log de log-loss por rodada de boosting no MLflow (opcional, custo baixo).

        Reaproveita o `eval_set` nativo do XGBoost -- a mesma passada que o
        boosting já faz internamente a cada rodada, sem nenhuma avaliação de
        ranking extra. Não influencia o treino (sem `early_stopping_rounds`);
        serve só para registrar a curva de log-loss. `test_feat`/`all_items`
        não são usados aqui -- fazem parte da mesma assinatura de
        `NeuralMLPModel.set_periodic_eval`, para `run_hyperparameter_search`
        poder chamar qualquer modelo de forma uniforme.
        """
        self._eval_data = (val_feat[feature_columns], val_feat["label"])
        return self

    def _fit_estimator(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Treina o XGBoost, logando log-loss por rodada se `set_periodic_eval` foi chamado."""
        if self._eval_data is not None and mlflow.active_run() is not None:
            X_val, y_val = self._eval_data
            self._clf.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
            self._log_eval_history()
        else:
            self._clf.fit(X, y)

    def _log_eval_history(self) -> None:
        """Loga a curva de log-loss (por rodada de boosting) já coletada pelo `eval_set`."""
        for data_name, metrics in self._clf.evals_result().items():
            for metric_name, values in metrics.items():
                for step, value in enumerate(values):
                    mlflow.log_metric(
                        f"{data_name}_{metric_name}", float(value), step=step
                    )

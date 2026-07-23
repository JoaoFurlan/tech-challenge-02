"""LightGBM sobre as features tabulares (`fe_v4`)."""

from typing import Self

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from recsys_ecommerce.models.tabular_classifier import TabularClassifierModel

# Sem n_jobs fixo, o LightGBM resolve o numero de threads sozinho -- alem do
# UserWarning de deteccao de nucleos fisicos no Windows (o motivo original
# deste fix), o numero de threads tambem muda conforme a maquina, e a ordem
# de agregacao ponto-flutuante dos histogramas de boosting muda junto (mesmo
# problema do XGBoost, ver xgboost_model.py). `os.cpu_count()` NAO resolve
# isso -- e exatamente o valor que difere entre maquinas. Só uma CONSTANTE
# fixa (a mesma em qualquer ambiente) elimina essa fonte de não-determinismo.
N_JOBS = 4


class LightGBMModel(TabularClassifierModel):
    """Gradient boosting (LightGBM), injetado em `TabularClassifierModel`."""

    def __init__(self, **kwargs: object) -> None:
        """Inicializa o modelo.

        Args:
            **kwargs: Repassados a `lightgbm.LGBMClassifier` (ex.:
                `num_leaves`, `max_depth`, `learning_rate`, `n_estimators`,
                `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`,
                `random_state`).
        """
        kwargs.setdefault("verbosity", -1)
        kwargs.setdefault("n_jobs", N_JOBS)
        # LGBMClassifier declara parametros tipados individualmente nos stubs;
        # **kwargs generico (o mesmo padrao dos outros modelos) nao type-checka
        # contra isso, mas e valido em runtime.
        super().__init__(LGBMClassifier(**kwargs))  # type: ignore[arg-type]
        self._eval_data: tuple[pd.DataFrame, pd.Series] | None = None

    def set_periodic_eval(
        self,
        val_feat: pd.DataFrame,
        test_feat: pd.DataFrame,
        feature_columns: list[str],
        all_items: np.ndarray,
    ) -> Self:
        """Liga o log de log-loss por rodada de boosting no MLflow (opcional, custo baixo).

        Reaproveita a avaliação nativa do LightGBM -- a mesma passada que o
        boosting já faz internamente a cada rodada, sem nenhuma avaliação de
        ranking extra. Não influencia o treino (sem early stopping); serve só
        para registrar a curva de log-loss. `test_feat`/`all_items` não são
        usados aqui -- fazem parte da mesma assinatura de
        `NeuralMLPModel.set_periodic_eval`, para `run_hyperparameter_search`
        poder chamar qualquer modelo de forma uniforme.
        """
        self._eval_data = (val_feat[feature_columns], val_feat["label"])
        return self

    def _fit_estimator(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Treina o LightGBM, logando log-loss por rodada se `set_periodic_eval` foi chamado."""
        if self._eval_data is not None and mlflow.active_run() is not None:
            X_val, y_val = self._eval_data
            # eval_set esta deprecado desde o lightgbm 4.7 em favor de eval_X/eval_y
            # (par unico, nao lista de tuplas).
            self._clf.fit(X, y, eval_X=X_val, eval_y=y_val, eval_metric="logloss")
            self._log_eval_history()
        else:
            self._clf.fit(X, y)

    def _log_eval_history(self) -> None:
        """Loga a curva de log-loss (por rodada de boosting) já coletada pela avaliação."""
        for data_name, metrics in self._clf.evals_result_.items():
            for metric_name, values in metrics.items():
                for step, value in enumerate(values):
                    mlflow.log_metric(
                        f"{data_name}_{metric_name}", float(value), step=step
                    )

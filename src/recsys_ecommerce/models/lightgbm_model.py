"""LightGBM sobre as features tabulares (`fe_v4`)."""

from lightgbm import LGBMClassifier

from recsys_ecommerce.models.tabular_classifier import TabularClassifierModel


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
        # LGBMClassifier declara parametros tipados individualmente nos stubs;
        # **kwargs generico (o mesmo padrao dos outros modelos) nao type-checka
        # contra isso, mas e valido em runtime.
        super().__init__(LGBMClassifier(**kwargs))  # type: ignore[arg-type]

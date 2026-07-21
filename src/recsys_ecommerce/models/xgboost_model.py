"""XGBoost sobre as features tabulares (`fe_v4`)."""

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

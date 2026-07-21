"""Baseline de regressão logística — o modelo tabular vencedor no sandbox (`ndcg@10 = 0.6471`)."""

from sklearn.linear_model import LogisticRegression

from recsys_ecommerce.models.tabular_classifier import TabularClassifierModel


class LogisticRegressionBaseline(TabularClassifierModel):
    """Regressão logística sobre as features tabulares (`fe_v4`), sem padronização.

    Surpreendentemente, o modelo mais simples testado no sandbox — venceu
    árvore de decisão, XGBoost e LightGBM tunados.
    """

    def __init__(self, **kwargs: object) -> None:
        """Inicializa o baseline.

        Args:
            **kwargs: Repassados a `sklearn.linear_model.LogisticRegression`
                (ex.: `C`, `solver`, `max_iter`, `random_state`).
        """
        super().__init__(LogisticRegression(**kwargs))

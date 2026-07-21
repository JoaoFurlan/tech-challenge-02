"""Árvore de decisão sobre as features tabulares (`fe_v4`)."""

from sklearn.tree import DecisionTreeClassifier

from recsys_ecommerce.models.tabular_classifier import TabularClassifierModel


class DecisionTreeModel(TabularClassifierModel):
    """Árvore de decisão sklearn, injetada em `TabularClassifierModel`."""

    def __init__(self, **kwargs: object) -> None:
        """Inicializa o modelo.

        Args:
            **kwargs: Repassados a `sklearn.tree.DecisionTreeClassifier`
                (ex.: `max_depth`, `min_samples_split`, `min_samples_leaf`,
                `random_state`).
        """
        super().__init__(DecisionTreeClassifier(**kwargs))

"""Etapa do pipeline DVC: treina a árvore de decisão (tuned).

Uso:
    uv run python scripts/pipeline/train_decision_tree.py
"""

from _common import run_tuned

from recsys_ecommerce.config import load_training_config, settings
from recsys_ecommerce.models.decision_tree_model import DecisionTreeModel


def main() -> None:
    """Ponto de entrada do estágio `train_decision_tree` do `dvc.yaml`."""
    cfg = load_training_config()
    tuned = DecisionTreeModel(
        max_depth=cfg.decision_tree_max_depth,
        min_samples_split=cfg.decision_tree_min_samples_split,
        min_samples_leaf=cfg.decision_tree_min_samples_leaf,
        random_state=settings.random_seed,
    )
    run_tuned("decision_tree", tuned)


if __name__ == "__main__":
    main()

"""Etapa do pipeline DVC: treina a regressão logística (tuned).

Uso:
    uv run python scripts/pipeline/train_baseline.py
"""

from _common import run_tuned

from recsys_ecommerce.config import load_training_config, settings
from recsys_ecommerce.models.sklearn_baseline import LogisticRegressionBaseline


def main() -> None:
    """Ponto de entrada do estágio `train_baseline` do `dvc.yaml`."""
    cfg = load_training_config()
    tuned = LogisticRegressionBaseline(
        C=cfg.logreg_c,
        solver=cfg.logreg_solver,
        max_iter=cfg.logreg_max_iter,
        random_state=settings.random_seed,
    )
    run_tuned("logreg", tuned)


if __name__ == "__main__":
    main()

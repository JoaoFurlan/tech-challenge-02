"""Etapa do pipeline DVC: treina a regressão logística (baseline + tuned).

Uso:
    uv run python scripts/pipeline/train_baseline.py
"""

from _common import run_baseline_and_tuned

from recsys_ecommerce.config import load_training_config, settings
from recsys_ecommerce.models.sklearn_baseline import LogisticRegressionBaseline


def main() -> None:
    """Ponto de entrada do estágio `train_baseline` do `dvc.yaml`."""
    cfg = load_training_config()
    baseline = LogisticRegressionBaseline(
        max_iter=1000, random_state=settings.random_seed
    )
    tuned = LogisticRegressionBaseline(
        C=cfg.logreg_c,
        solver=cfg.logreg_solver,
        max_iter=cfg.logreg_max_iter,
        random_state=settings.random_seed,
    )
    run_baseline_and_tuned("logreg", baseline, tuned)


if __name__ == "__main__":
    main()

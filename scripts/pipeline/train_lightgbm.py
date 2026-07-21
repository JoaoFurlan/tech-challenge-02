"""Etapa do pipeline DVC: treina o LightGBM (baseline + tuned).

Uso:
    uv run python scripts/pipeline/train_lightgbm.py
"""

from _common import run_baseline_and_tuned

from recsys_ecommerce.config import load_training_config, settings
from recsys_ecommerce.models.lightgbm_model import LightGBMModel


def main() -> None:
    """Ponto de entrada do estágio `train_lightgbm` do `dvc.yaml`."""
    cfg = load_training_config()
    baseline = LightGBMModel(n_estimators=100, random_state=settings.random_seed)
    tuned = LightGBMModel(
        num_leaves=cfg.lightgbm_num_leaves,
        max_depth=cfg.lightgbm_max_depth,
        learning_rate=cfg.lightgbm_learning_rate,
        n_estimators=cfg.lightgbm_n_estimators,
        subsample=cfg.lightgbm_subsample,
        colsample_bytree=cfg.lightgbm_colsample_bytree,
        reg_alpha=cfg.lightgbm_reg_alpha,
        reg_lambda=cfg.lightgbm_reg_lambda,
        random_state=settings.random_seed,
    )
    run_baseline_and_tuned("lightgbm", baseline, tuned)


if __name__ == "__main__":
    main()

"""Etapa do pipeline DVC: treina o XGBoost (tuned).

Uso:
    uv run python scripts/pipeline/train_xgboost.py
"""

from _common import run_tuned

from recsys_ecommerce.config import load_training_config, settings
from recsys_ecommerce.models.xgboost_model import XGBoostModel


def main() -> None:
    """Ponto de entrada do estágio `train_xgboost` do `dvc.yaml`."""
    cfg = load_training_config()
    tuned = XGBoostModel(
        max_depth=cfg.xgboost_max_depth,
        learning_rate=cfg.xgboost_learning_rate,
        n_estimators=cfg.xgboost_n_estimators,
        subsample=cfg.xgboost_subsample,
        colsample_bytree=cfg.xgboost_colsample_bytree,
        reg_alpha=cfg.xgboost_reg_alpha,
        reg_lambda=cfg.xgboost_reg_lambda,
        random_state=settings.random_seed,
    )
    run_tuned("xgboost", tuned)


if __name__ == "__main__":
    main()

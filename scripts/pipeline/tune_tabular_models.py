"""Tunagem de verdade dos 4 modelos tabulares: script standalone e resumível, fora do `dvc.yaml`.

Complementa (não substitui) o par `baseline`/`tuned` já logado por
`train_baseline.py`/`train_decision_tree.py`/`train_xgboost.py`/
`train_lightgbm.py` -- aqueles usam os hiperparâmetros já vencedores
encontrados no sandbox; este script faz uma busca aleatória de verdade,
de novo, sobre `model-selection`, tag `trial_type="tuned"`. `register_best_trial`
varre TODOS os trials de cada família (os 2 antigos + os novos daqui) e
registra o de melhor `test_ndcg` -- nada do que já existe é perdido ou
sobrescrito, só ganha mais candidatos para comparar.

Resumível de verdade (mesma mecânica de `tune_neural_mlp.py`): interrompa a
qualquer momento e rode de novo -- os trials já logados são pulados.

Uso:
    uv run python scripts/pipeline/tune_tabular_models.py
"""

import sys
from typing import Any

import mlflow
from _common import EXPERIMENT_NAME, load_featured_tables, load_winning_feature_set

from recsys_ecommerce.config import load_training_config, settings
from recsys_ecommerce.models.decision_tree_model import DecisionTreeModel
from recsys_ecommerce.models.lightgbm_model import LightGBMModel
from recsys_ecommerce.models.sklearn_baseline import LogisticRegressionBaseline
from recsys_ecommerce.models.xgboost_model import XGBoostModel
from recsys_ecommerce.tracking.mlflow_organization import (
    register_best_trial,
    run_hyperparameter_search,
)

if (reconfigure := getattr(sys.stdout, "reconfigure", None)) is not None:
    reconfigure(encoding="utf-8")

N_TRIALS = 12

SEARCH_SPECS: list[tuple[str, type, dict[str, list[Any]], dict[str, Any]]] = [
    (
        "logreg",
        LogisticRegressionBaseline,
        {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]},
        {"solver": "lbfgs", "max_iter": 1000, "random_state": settings.random_seed},
    ),
    (
        "decision_tree",
        DecisionTreeModel,
        {
            "max_depth": [3, 4, 5, 6, 7, 8, 10],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 5, 10, 20],
        },
        {"random_state": settings.random_seed},
    ),
    (
        "xgboost",
        XGBoostModel,
        {
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.005, 0.01, 0.02, 0.05, 0.1],
            "n_estimators": [50, 100, 150, 200],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.5, 0.6, 0.8, 1.0],
            "reg_alpha": [0.0, 0.01, 0.1],
            "reg_lambda": [0.0, 0.1, 0.5, 1.0],
        },
        {"random_state": settings.random_seed},
    ),
    (
        "lightgbm",
        LightGBMModel,
        {
            "num_leaves": [7, 15, 31, 63],
            "max_depth": [-1, 4, 6, 8, 10],
            "learning_rate": [0.005, 0.01, 0.02, 0.05, 0.1],
            "n_estimators": [50, 100, 150, 200],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.5, 0.6, 0.8, 1.0],
            "reg_alpha": [0.0, 0.01, 0.1],
            "reg_lambda": [0.0, 0.1, 0.5, 1.0],
        },
        {"random_state": settings.random_seed},
    ),
]


def main() -> None:
    """Roda uma busca aleatória de 12 trials para cada um dos 4 modelos tabulares."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    feature_set = load_winning_feature_set()
    tables = load_featured_tables(feature_set)
    cfg = load_training_config()

    for model_family, model_class, search_space, fixed_params in SEARCH_SPECS:
        run_hyperparameter_search(
            search_name=f"{model_family}-{feature_set}",
            trial_type="tuned",
            model_class=model_class,
            search_space=search_space,
            n_trials=N_TRIALS,
            model_family=model_family,
            feature_set=feature_set,
            experiment_name=EXPERIMENT_NAME,
            train_feat=tables.train,
            val_feat=tables.val,
            test_feat=tables.test,
            train_eval_feat=tables.train_eval,
            feature_columns=tables.feature_columns,
            all_items=tables.all_items,
            fixed_params=fixed_params,
            seed=42,
        )
        register_best_trial(EXPERIMENT_NAME, model_family, cfg.registered_model_name)


if __name__ == "__main__":
    main()

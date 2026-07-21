"""Compara cada versão de feature engineering (`fe_v1`..`fe_v4`) x cada modelo tabular.

Script "pesado", standalone, fora do `dvc.yaml` (mesma categoria de
`tune_neural_mlp.py`/`tune_tabular_models.py`) -- roda sob demanda, não faz
parte do caminho obrigatório/rápido (`dvc repro`). Hiperparâmetros default
(não tunados) em todos os modelos -- o ponto é isolar o efeito da FEATURE
ENGINEERING, não o de hiperparâmetros (isso é `model-selection`, feito
depois, só sobre o `feature_set` vencedor). Runs flat, tags
`model_family`/`feature_set`, sem artefato de modelo (nenhuma run aqui é
candidata a produção).

Este script só recomenda um vencedor (grava `reports/feature_engineering_winner.json`
como relatório informativo) -- não decide nada automaticamente. Para adotar
um vencedor de verdade (usado por `feature_eng.py` e `model-selection`), rode
`scripts/pipeline/promote_feature_set.py <nome>` manualmente, depois de revisar
os números aqui.

Uso:
    uv run python scripts/experiments/run_fe_comparison.py
"""

import json
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from recsys_ecommerce.config import settings
from recsys_ecommerce.evaluation.metrics import evaluate_model
from recsys_ecommerce.evaluation.plots import (
    plot_feature_importance,
    plot_metrics_across_splits,
)
from recsys_ecommerce.features.basic import FEATURE_COLUMNS_V1
from recsys_ecommerce.features.pipeline import (
    FEATURE_COLUMNS_V3,
    FEATURE_COLUMNS_V4,
    InteractionTables,
    build_fe_v1,
    build_fe_v2_variants,
    build_fe_v3,
    build_fe_v4,
)
from recsys_ecommerce.models.tabular_classifier import TabularClassifierModel
from recsys_ecommerce.tracking.mlflow_organization import find_run_id

if (reconfigure := getattr(sys.stdout, "reconfigure", None)) is not None:
    reconfigure(encoding="utf-8")

EXPERIMENT_NAME = "feature-engineering"
REPORTS_DIR = Path("reports")
FEATURE_COLUMNS_V2 = [*FEATURE_COLUMNS_V1, "item_covisitation_score"]

# Hiperparâmetros default (não tunados) -- o ponto desta comparação é isolar
# o efeito da FEATURE ENGINEERING, não o de hiperparâmetros (isso é
# `model-selection`, feito depois, só sobre a `feature_set` vencedora).
MODEL_SPECS: list[tuple[str, Callable[[], TabularClassifierModel]]] = [
    (
        "logreg",
        lambda: TabularClassifierModel(
            LogisticRegression(max_iter=1000, random_state=42)
        ),
    ),
    (
        "decision_tree",
        lambda: TabularClassifierModel(
            DecisionTreeClassifier(max_depth=8, random_state=42)
        ),
    ),
    (
        "xgboost",
        lambda: TabularClassifierModel(
            XGBClassifier(
                n_estimators=100, max_depth=6, eval_metric="logloss", random_state=42
            )
        ),
    ),
    (
        "lightgbm",
        lambda: TabularClassifierModel(
            LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1)
        ),
    ),
]


def _run_one(
    feature_set_name: str,
    model_name: str,
    make_model: Callable[[], TabularClassifierModel],
    tables: dict[str, pd.DataFrame],
    feature_columns: list[str],
    all_items: np.ndarray,
) -> float:
    """Loga (ou retoma) `{feature_set_name}__{model_name}`. Retorna `test_ndcg`."""
    run_name = f"{feature_set_name}__{model_name}"
    existing_id = find_run_id(EXPERIMENT_NAME, run_name)
    if existing_id is not None:
        print(f"{run_name} já logado, pulando.")
        return float(
            mlflow.MlflowClient().get_run(existing_id).data.metrics["test_ndcg"]
        )

    X_train, y_train = tables["train"][feature_columns], tables["train"]["label"]
    model = make_model()
    model.fit(X_train, y_train)

    train_metrics = evaluate_model(
        model, tables["train_eval"], feature_columns, all_items
    )
    val_metrics = evaluate_model(model, tables["val"], feature_columns, all_items)
    test_metrics = evaluate_model(model, tables["test"], feature_columns, all_items)

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"model_family": model_name, "feature_set": feature_set_name})
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        fig_splits = plot_metrics_across_splits(
            train_metrics, val_metrics, test_metrics, run_name
        )
        mlflow.log_figure(fig_splits, "metrics_by_split.png")
        plt.close(fig_splits)

        fig_importance = plot_feature_importance(
            model.underlying_estimator, feature_columns, model_name
        )
        mlflow.log_figure(fig_importance, "feature_importance.png")
        plt.close(fig_importance)

    print(f"  {run_name}: test_ndcg={test_metrics['ndcg']:.4f}")
    return test_metrics["ndcg"]


def _run_batch(
    feature_set_name: str,
    tables: dict[str, pd.DataFrame],
    feature_columns: list[str],
    all_items: np.ndarray,
) -> dict[str, float]:
    """Roda os 4 modelos tabulares sobre uma versão de FE. Retorna `{model_name: test_ndcg}`."""
    return {
        model_name: _run_one(
            feature_set_name, model_name, make_model, tables, feature_columns, all_items
        )
        for model_name, make_model in MODEL_SPECS
    }


def main() -> None:
    """Roda a comparação completa: `fe_v1`, `fe_v2.0`-`2.4`, `fe_v3`, `fe_v4` x 4 modelos."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    client = mlflow.MlflowClient()
    client.set_experiment_tag(
        experiment.experiment_id,
        "mlflow.note.content",
        "Comparação única `model_family` x `feature_set`, hiperparâmetros default -- "
        "objetivo: escolher o `feature_set` vencedor antes de qualquer tuning. Runs "
        "flat, sem artefato de modelo (nenhuma run aqui é candidata a produção). Ver "
        "`model-selection` para o que vem depois -- consome "
        "`reports/feature_engineering_winner.json`.",
    )

    tables = InteractionTables.load(settings.data_dir)
    all_scores: dict[str, dict[str, float]] = {}

    all_scores["fe_v1"] = _run_batch(
        "fe_v1", build_fe_v1(tables), FEATURE_COLUMNS_V1, tables.all_items
    )

    raw_events = pd.read_csv(settings.data_dir / "raw" / "events.csv")
    fe_v2_variants = build_fe_v2_variants(tables, raw_events)
    for variant_name, variant_tables in fe_v2_variants.items():
        all_scores[variant_name] = _run_batch(
            variant_name, variant_tables, FEATURE_COLUMNS_V2, tables.all_items
        )

    all_scores["fe_v3"] = _run_batch(
        "fe_v3",
        build_fe_v3(tables, settings.data_dir),
        FEATURE_COLUMNS_V3,
        tables.all_items,
    )
    all_scores["fe_v4"] = _run_batch(
        "fe_v4",
        build_fe_v4(tables, settings.data_dir),
        FEATURE_COLUMNS_V4,
        tables.all_items,
    )

    winner = max(
        all_scores, key=lambda fs: float(np.mean(list(all_scores[fs].values())))
    )
    winner_mean_ndcg = float(np.mean(list(all_scores[winner].values())))
    print(f"feature_set vencedor: {winner} (ndcg médio={winner_mean_ndcg:.4f})")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "feature_engineering_winner.json").write_text(
        json.dumps({"winner": winner, "scores": all_scores}, indent=2)
    )


if __name__ == "__main__":
    main()

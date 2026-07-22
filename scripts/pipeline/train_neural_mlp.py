"""Etapa do pipeline DVC: treina o MLP (tuned).

Mesmo padrão dos modelos tabulares (`train_lightgbm.py` etc.): um trial fixo
por execução -- `tuned`, os vencedores já congelados em `configs/model.yaml`
(encontrados por `tune_neural_mlp.py`, fora do `dvc.yaml`). Não reaproveita
`_common.run_tuned` porque o MLP precisa de tratamento próprio que os
modelos tabulares não têm (curva de perda em vez de importância de
features, reavaliação periódica via `set_periodic_eval`).

Uso:
    uv run python scripts/pipeline/train_neural_mlp.py
"""

import json
import sys

import matplotlib.pyplot as plt
import mlflow
from _common import EXPERIMENT_NAME, REPORTS_DIR, load_winning_feature_set
from matplotlib.figure import Figure

from recsys_ecommerce.config import load_training_config, settings
from recsys_ecommerce.evaluation.metrics import evaluate_model
from recsys_ecommerce.evaluation.plots import plot_metrics_across_splits
from recsys_ecommerce.features.pipeline import FeaturedTables
from recsys_ecommerce.models.neural_mlp import NeuralMLPModel
from recsys_ecommerce.tracking.mlflow_organization import (
    find_run_id,
    log_model_artifact,
    register_best_trial,
)

if (reconfigure := getattr(sys.stdout, "reconfigure", None)) is not None:
    reconfigure(encoding="utf-8")


def _plot_loss_curve(model: NeuralMLPModel, run_name: str) -> Figure:
    history = model.underlying_estimator.loss_history_
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train"], label="train")
    ax.plot(history["early_stop"], label="early-stop (interno)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title(f"Curva de perda -- {run_name}")
    ax.legend()
    plt.tight_layout()
    return fig


def _train_and_log_trial(
    trial_type: str, model: NeuralMLPModel, feature_set: str, tables: FeaturedTables
) -> float:
    """Loga (ou retoma) um trial flat `mlp-{trial_type}`. Retorna o `test_ndcg`.

    Idempotente: pula o treino se a run já existir, relendo `test_ndcg` do
    que já está logado -- mesmo contrato de `_common.train_and_log_trial`.
    """
    run_name = f"mlp-{trial_type}"
    existing_id = find_run_id(EXPERIMENT_NAME, run_name)
    if existing_id is not None:
        print(f"{run_name} já logado, pulando.")
        return float(
            mlflow.MlflowClient().get_run(existing_id).data.metrics["test_ndcg"]
        )

    X_train = tables.train[tables.feature_columns]
    y_train = tables.train["label"]

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "model_family": "mlp",
                "feature_set": feature_set,
                "trial_type": trial_type,
            }
        )
        mlflow.log_params(model.get_params())

        model.set_periodic_eval(
            tables.val, tables.test, tables.feature_columns, tables.all_items
        )
        model.fit(X_train, y_train)

        train_metrics = evaluate_model(
            model, tables.train_eval, tables.feature_columns, tables.all_items
        )
        val_metrics = evaluate_model(
            model, tables.val, tables.feature_columns, tables.all_items
        )
        test_metrics = evaluate_model(
            model, tables.test, tables.feature_columns, tables.all_items
        )

        mlflow.log_metric(
            "n_epochs_trained", model.underlying_estimator.n_epochs_trained_
        )
        mlflow.log_metric("pos_weight", model.underlying_estimator.pos_weight_)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        fig_loss = _plot_loss_curve(model, run_name)
        mlflow.log_figure(fig_loss, "loss_curve.png")
        plt.close(fig_loss)

        fig_splits = plot_metrics_across_splits(
            train_metrics, val_metrics, test_metrics, run_name
        )
        mlflow.log_figure(fig_splits, "metrics_by_split.png")
        plt.close(fig_splits)

        log_model_artifact(model)

    test_ndcg = test_metrics["ndcg"]
    print(f"{run_name} -- test_ndcg={test_ndcg:.4f}")
    return test_ndcg


def main() -> None:
    """Ponto de entrada do estágio `train_neural_mlp` do `dvc.yaml`."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    cfg = load_training_config()
    feature_set = load_winning_feature_set()
    tables = FeaturedTables.load(settings.data_dir, feature_set)

    tuned = NeuralMLPModel(
        hidden_dims=tuple(cfg.mlp_hidden_dims),
        dropout=cfg.mlp_dropout,
        lr=cfg.mlp_lr,
        weight_decay=cfg.mlp_weight_decay,
        batch_size=cfg.mlp_batch_size,
        max_epochs=cfg.mlp_max_epochs,
        patience=cfg.mlp_patience,
        weighted_loss=cfg.mlp_weighted_loss,
        eval_every_n_epochs=cfg.mlp_eval_every_n_epochs,
        seed=settings.random_seed,
    )

    tuned_ndcg = _train_and_log_trial("tuned", tuned, feature_set, tables)

    register_best_trial(EXPERIMENT_NAME, "mlp", cfg.registered_model_name)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "mlp_metrics.json").write_text(
        json.dumps({"test_ndcg": tuned_ndcg}, indent=2)
    )


if __name__ == "__main__":
    main()

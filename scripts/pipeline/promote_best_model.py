"""Última etapa do pipeline DVC: promove o melhor modelo a Production.

Compara os candidatos já registrados no Model Registry por
`register_best_trial` (um por família de modelo -- `logreg`,
`decision_tree`, `xgboost`, `lightgbm`, `mlp`) e promove o de maior
`test_ndcg` a `Production`. Não retreina nada, não abre nenhuma run própria
-- a comparação usa só as tags já gravadas em cada versão registrada
(`model_family`, `test_ndcg`).

Uso:
    uv run python scripts/pipeline/promote_best_model.py
"""

import json
import sys
from pathlib import Path

import mlflow

from recsys_ecommerce.config import load_training_config, settings

if (reconfigure := getattr(sys.stdout, "reconfigure", None)) is not None:
    reconfigure(encoding="utf-8")

REPORTS_DIR = Path("reports")


def main() -> None:
    """Ponto de entrada do estágio `promote_best_model` do `dvc.yaml`."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    cfg = load_training_config()
    client = mlflow.MlflowClient()

    versions = client.search_model_versions(f"name='{cfg.registered_model_name}'")
    if not versions:
        print("Nenhum modelo registrado ainda -- rode os estágios train_* primeiro.")
        return

    best = max(versions, key=lambda v: float(v.tags.get("test_ndcg", "-inf")))
    client.transition_model_version_stage(
        name=cfg.registered_model_name,
        version=best.version,
        stage="Production",
        archive_existing_versions=True,
    )

    rows = [
        {
            "model_family": v.tags.get("model_family"),
            "version": v.version,
            "test_ndcg": float(v.tags["test_ndcg"]),
        }
        for v in versions
        if "test_ndcg" in v.tags
    ]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(rows, indent=2))

    print(
        f"Promovido a Production: {best.tags.get('model_family')} "
        f"(versão {best.version}, test_ndcg={best.tags['test_ndcg']})"
    )


if __name__ == "__main__":
    main()

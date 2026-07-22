"""Última etapa do pipeline DVC: promove o melhor modelo (alias `production`).

Compara os candidatos já registrados no Model Registry por
`register_best_trial` (um por família de modelo -- `logreg`,
`decision_tree`, `xgboost`, `lightgbm`, `mlp`) e promove o de maior
`test_ndcg`, apontando o alias `production` pra ele. Não retreina nada, não
abre nenhuma run própria -- a comparação usa só as tags já gravadas em cada
versão registrada (`model_family`, `test_ndcg`).

Usa aliases (`set_registered_model_alias`), não o antigo conceito de Stages
(`transition_model_version_stage`) -- API deprecada desde o MLflow 2.9,
a ser removida numa versão futura. Reatribuir o mesmo alias a uma nova
versão já tira automaticamente da anterior (um alias aponta pra só uma
versão por vez), sem precisar de um "archive_existing_versions" explícito.

Uso:
    uv run python scripts/pipeline/promote_best_model.py
"""

import json
import sys
from pathlib import Path

import mlflow
from _common import log_stage_timing

from recsys_ecommerce.config import load_training_config, settings

if (reconfigure := getattr(sys.stdout, "reconfigure", None)) is not None:
    reconfigure(encoding="utf-8")

REPORTS_DIR = Path("reports")
PRODUCTION_ALIAS = "production"


def main() -> None:
    """Ponto de entrada do estágio `promote_best_model` do `dvc.yaml`."""
    with log_stage_timing("promote_best_model"):
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        cfg = load_training_config()
        client = mlflow.MlflowClient()

        versions = client.search_model_versions(f"name='{cfg.registered_model_name}'")
        if not versions:
            print(
                "Nenhum modelo registrado ainda -- rode os estágios train_* primeiro."
            )
            return

        best = max(versions, key=lambda v: float(v.tags.get("test_ndcg", "-inf")))
        client.set_registered_model_alias(
            name=cfg.registered_model_name,
            alias=PRODUCTION_ALIAS,
            version=best.version,
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
            f"Promovido (alias '{PRODUCTION_ALIAS}'): {best.tags.get('model_family')} "
            f"(versão {best.version}, test_ndcg={best.tags['test_ndcg']})"
        )


if __name__ == "__main__":
    main()

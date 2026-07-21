"""Troca o feature set vencedor usado em produção -- decisão manual, propagação automática.

A decisão de QUAL feature set vencer é sempre manual (revise os números de
`scripts/experiments/run_fe_comparison.py` antes de rodar isto). O que este
script automatiza é a propagação dessa decisão para todo lugar que precisa
saber -- sem isto, seria preciso editar `configs/model.yaml` e `dvc.yaml` à
mão, em mais de um lugar cada.

Atualiza, em conjunto:

1. `configs/model.yaml:winning_feature_set` -- o que `feature_eng.py` e os
   scripts de `model-selection` de fato leem (ver `_common.load_winning_feature_set`).
2. `dvc.yaml` -- os caminhos `data/processed/{antigo}` -> `data/processed/{novo}`.
3. MLflow -- tag `promoted_feature_set` no experimento `feature-engineering`,
   só para visibilidade na UI (não é a fonte de verdade -- se o experimento
   ainda não existe, ou se o tracking server estiver fora, este passo é
   pulado sem erro; os arquivos locais são o que realmente importa).

Uso:
    uv run python scripts/pipeline/promote_feature_set.py fe_v5
"""

import re
import sys
from pathlib import Path

import mlflow

from recsys_ecommerce.config import load_training_config, settings
from recsys_ecommerce.features.pipeline import FEATURE_SET_BUILDERS

MODEL_YAML_PATH = Path("configs/model.yaml")
DVC_YAML_PATH = Path("dvc.yaml")
FE_EXPERIMENT_NAME = "feature-engineering"


def _update_model_yaml(new_feature_set: str) -> None:
    """Reescreve a linha `winning_feature_set:` em `configs/model.yaml`, preservando o resto."""
    text = MODEL_YAML_PATH.read_text(encoding="utf-8")
    updated, n = re.subn(
        r"^winning_feature_set:\s*\S+",
        f"winning_feature_set: {new_feature_set}",
        text,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise RuntimeError(
            f"esperava encontrar exatamente 1 'winning_feature_set:' em "
            f"{MODEL_YAML_PATH}, achei {n}"
        )
    MODEL_YAML_PATH.write_text(updated, encoding="utf-8")


def _update_dvc_yaml(old_feature_set: str, new_feature_set: str) -> None:
    """Troca `data/processed/{antigo}` por `data/processed/{novo}` em todo `dvc.yaml`."""
    text = DVC_YAML_PATH.read_text(encoding="utf-8")
    old_path, new_path = (
        f"data/processed/{old_feature_set}",
        f"data/processed/{new_feature_set}",
    )
    updated = text.replace(old_path, new_path)
    if updated == text:
        raise RuntimeError(
            f"nenhuma ocorrência de '{old_path}' encontrada em {DVC_YAML_PATH}"
        )
    DVC_YAML_PATH.write_text(updated, encoding="utf-8")


def _tag_mlflow(new_feature_set: str) -> None:
    """Marca o feature set promovido no experimento `feature-engineering`, só para visibilidade."""
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        experiment = mlflow.get_experiment_by_name(FE_EXPERIMENT_NAME)
        if experiment is None:
            print(
                f"experimento '{FE_EXPERIMENT_NAME}' não existe ainda -- pulando tag no MLflow."
            )
            return
        mlflow.MlflowClient().set_experiment_tag(
            experiment.experiment_id, "promoted_feature_set", new_feature_set
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"não consegui marcar no MLflow ({exc}) -- seguindo sem isso, não é a fonte de verdade."
        )


def main() -> None:
    """Ponto de entrada: `uv run python scripts/pipeline/promote_feature_set.py <feature_set>`."""
    if len(sys.argv) != 2:
        raise SystemExit(
            "uso: uv run python scripts/pipeline/promote_feature_set.py <feature_set>"
        )
    new_feature_set = sys.argv[1]
    if new_feature_set not in FEATURE_SET_BUILDERS:
        known = ", ".join(sorted(FEATURE_SET_BUILDERS))
        raise SystemExit(
            f"'{new_feature_set}' não está em FEATURE_SET_BUILDERS (conhecidos: {known})"
        )

    old_feature_set = load_training_config().winning_feature_set
    if old_feature_set == new_feature_set:
        print(f"'{new_feature_set}' já é o feature_set vencedor -- nada a fazer.")
        return

    _update_model_yaml(new_feature_set)
    _update_dvc_yaml(old_feature_set, new_feature_set)
    _tag_mlflow(new_feature_set)

    print(f"feature_set vencedor trocado: {old_feature_set} -> {new_feature_set}")
    print(
        "Próximo passo: `dvc repro` vai detectar a mudança e re-treinar "
        "tudo sobre o novo feature set."
    )


if __name__ == "__main__":
    main()

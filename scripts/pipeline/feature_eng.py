"""Segunda etapa do pipeline DVC: engenharia de features (a versão vencedora).

Lê as tabelas gravadas por `preprocess` e persiste o feature set marcado em
`configs/model.yaml:winning_feature_set` -- essa escolha é manual (revisada
via `scripts/experiments/run_fe_comparison.py`) e trocada com
`scripts/pipeline/promote_feature_set.py`, nunca editada aqui.

Uso:
    uv run python scripts/pipeline/feature_eng.py
"""

import json
from pathlib import Path

import pandas as pd
from _common import log_stage_timing

from recsys_ecommerce.config import load_training_config, settings
from recsys_ecommerce.features.pipeline import FEATURE_SET_BUILDERS, InteractionTables


def run_feature_eng(data_dir: Path, feature_set_name: str) -> dict[str, pd.DataFrame]:
    """Roda a engenharia de features vencedora e grava o resultado em `data/processed/{nome}/`.

    Args:
        data_dir: Diretório base de dados (contém `raw/` e `processed/`).
        feature_set_name: Nome do feature set a persistir (ex.: `"fe_v4"`),
            deve estar em `FEATURE_SET_BUILDERS`.

    Returns:
        As 4 tabelas com as features (`train`/`val`/`test`/`train_eval`).
    """
    build_fn, feature_columns = FEATURE_SET_BUILDERS[feature_set_name]
    tables = InteractionTables.load(data_dir)
    featured = build_fn(tables, data_dir)

    out_dir = data_dir / "processed" / feature_set_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in featured.items():
        table.to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / "feature_columns.json").write_text(json.dumps(feature_columns))
    pd.DataFrame({"itemid": tables.all_items}).to_parquet(
        out_dir / "all_items.parquet", index=False
    )

    print(f"{feature_set_name} gravado em {out_dir} -- colunas: {feature_columns}")
    for name, table in featured.items():
        print(f"  {name}: {len(table):,} linhas")
    return featured


def main() -> None:
    """Ponto de entrada do estágio `feature_eng` do `dvc.yaml`."""
    with log_stage_timing("feature_eng"):
        cfg = load_training_config()
        run_feature_eng(settings.data_dir, cfg.winning_feature_set)


if __name__ == "__main__":
    main()

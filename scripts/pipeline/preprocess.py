"""Primeira etapa do pipeline DVC: cold-start, split leave-one-out e negative sampling.

Lê `data/raw/events.csv`, aplica o filtro de cold-start (deduplicando antes,
para que `min_interactions` conte itens/usuários únicos, não eventos
repetidos), o split leave-one-out por usuário, e o negative sampling (1:4 no
treino, 1:99 na avaliação). Grava as tabelas resultantes em
`data/processed/interactions/`, consumidas pela etapa `feature_eng`.

Uso:
    uv run python scripts/pipeline/preprocess.py
"""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from _common import log_stage_timing

from recsys_ecommerce.config import load_training_config, settings
from recsys_ecommerce.preprocessing.interaction import (
    assign_loo_splits,
    build_labeled_table,
    deduplicate_interactions,
    filter_cold_start,
    sample_negatives,
)

TRAIN_EVAL_SAMPLE_SIZE = 50_000
OUTPUT_DIR_NAME = "interactions"


def _build_split_table(
    split_name: str,
    split_df: pd.DataFrame,
    positive_items_by_user: dict[int, set[int]],
    all_items: np.ndarray,
    n_neg_per_user: int,
    rng: np.random.Generator,
    seed: int,
) -> pd.DataFrame:
    """Monta a tabela rotulada (positivos + negativos amostrados) de um split."""
    positives = split_df.loc[
        split_df["split"] == split_name, ["visitorid", "itemid"]
    ].assign(label=1)
    negatives = sample_negatives(
        positives["visitorid"], positive_items_by_user, all_items, n_neg_per_user, rng
    )
    return build_labeled_table(positives, negatives, seed=seed)


def _build_train_eval_table(
    train_positives: pd.DataFrame,
    positive_items_by_user: dict[int, set[int]],
    all_items: np.ndarray,
    n_neg_per_user: int,
    sample_size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Diagnóstico de overfitting: 1 positivo por usuário (amostra de treino) + negativos.

    Ao contrário de val/test (exatamente 1 item por usuário, por design do
    leave-one-out), train tem muitos positivos por usuário. Para manter o
    mesmo formato "1 positivo + N negativos" que `evaluate_model` espera,
    amostra-se exatamente 1 positivo por usuário aqui.
    """
    eval_users = rng.choice(
        train_positives["visitorid"].unique(), size=sample_size, replace=False
    )
    eval_positives = (
        train_positives[train_positives["visitorid"].isin(eval_users)]
        .groupby("visitorid", as_index=False)
        .sample(n=1, random_state=5)
    )
    negatives = sample_negatives(
        eval_positives["visitorid"],
        positive_items_by_user,
        all_items,
        n_neg_per_user,
        rng,
    )
    return build_labeled_table(eval_positives, negatives, seed=14)


def run_preprocess(
    data_dir: Path,
    min_interactions: int,
    negative_sampling_ratio: int,
    eval_negative_samples: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Roda o pré-processamento completo e grava as tabelas em `data/processed/interactions/`.

    Args:
        data_dir: Diretório base de dados (contém `raw/` e onde `processed/`
            será criado).
        min_interactions: Limite mínimo de interações por usuário/item
            (filtro de cold-start).
        negative_sampling_ratio: Negativos por positivo no treino.
        eval_negative_samples: Negativos por positivo em val/test/train_eval.
        seed: Semente do gerador de números aleatórios.

    Returns:
        As tabelas geradas (`split_df`, `train_table`, `val_table`,
        `test_table`, `train_eval_table`, `all_items`), também gravadas em
        disco.
    """
    events = pd.read_csv(data_dir / "raw" / "events.csv")

    # Deduplicar ANTES de filtrar garante que min_interactions conta
    # usuários/itens únicos, não eventos (view/addtocart/transaction) repetidos.
    deduped_all_events = deduplicate_interactions(events)
    filtered = filter_cold_start(deduped_all_events, min_interactions)
    split_df = assign_loo_splits(filtered)  # já deduplicado, não precisa dedup de novo

    positive_items_by_user = cast(
        "dict[int, set[int]]",
        split_df.groupby("visitorid")["itemid"].apply(set).to_dict(),
    )
    all_items = split_df["itemid"].unique()
    rng = np.random.default_rng(seed)

    train_positives = split_df.loc[
        split_df["split"] == "train", ["visitorid", "itemid"]
    ].assign(label=1)
    train_table = _build_split_table(
        "train",
        split_df,
        positive_items_by_user,
        all_items,
        negative_sampling_ratio,
        rng,
        seed=10,
    )
    val_table = _build_split_table(
        "val",
        split_df,
        positive_items_by_user,
        all_items,
        eval_negative_samples,
        rng,
        seed=11,
    )
    test_table = _build_split_table(
        "test",
        split_df,
        positive_items_by_user,
        all_items,
        eval_negative_samples,
        rng,
        seed=12,
    )
    train_eval_table = _build_train_eval_table(
        train_positives,
        positive_items_by_user,
        all_items,
        eval_negative_samples,
        TRAIN_EVAL_SAMPLE_SIZE,
        rng,
    )

    tables = {
        "split_df": split_df,
        "train_table": train_table,
        "val_table": val_table,
        "test_table": test_table,
        "train_eval_table": train_eval_table,
        "all_items": pd.DataFrame({"itemid": all_items}),
    }
    out_dir = data_dir / "processed" / OUTPUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(out_dir / f"{name}.parquet", index=False)

    print(
        f"train: {len(train_table):,} | val: {len(val_table):,} | "
        f"test: {len(test_table):,} | train_eval: {len(train_eval_table):,}"
    )
    return tables


def main() -> None:
    """Ponto de entrada do estágio `preprocess` do `dvc.yaml`."""
    with log_stage_timing("preprocess"):
        cfg = load_training_config()
        run_preprocess(
            data_dir=settings.data_dir,
            min_interactions=cfg.min_interactions,
            negative_sampling_ratio=cfg.negative_sampling_ratio,
            eval_negative_samples=cfg.eval_negative_samples,
            seed=settings.random_seed,
        )


if __name__ == "__main__":
    main()

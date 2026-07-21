"""Orquestra as camadas de feature engineering (`fe_v1` a `fe_v4`) sobre as tabelas do `preprocess`.

`FEATURE_SET_BUILDERS` registra as versões elegíveis para produção --
`scripts/pipeline/feature_eng.py` persiste a que estiver marcada em
`configs/model.yaml:winning_feature_set` (trocada via
`scripts/pipeline/promote_feature_set.py`), não mais fixo em `fe_v4`.
`build_fe_v2_variants` fica fora do registro -- suas 5 variantes existem só
para o script de comparação (`scripts/experiments/run_fe_comparison.py`)
recriar a mesma superfície de comparação que já existe no MLflow do sandbox.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from recsys_ecommerce.features.basic import FEATURE_COLUMNS_V1, BasicFeatures
from recsys_ecommerce.features.category import (
    CategoryAffinityArtifacts,
    build_item_to_parent_category,
    compute_item_relative_popularity,
    load_category_parent_map,
    load_item_categories,
)
from recsys_ecommerce.features.covisitation import (
    CovisitationArtifacts,
    compute_covisitation_scores,
)

FEATURE_COLUMNS_V3 = [
    *FEATURE_COLUMNS_V1,
    "item_covisitation_cosine",
    "category_affinity",
]
FEATURE_COLUMNS_V4 = [
    *FEATURE_COLUMNS_V3,
    "parent_category_affinity",
    "item_relative_popularity",
]

_SPLIT_NAMES = ("train", "val", "test", "train_eval")


@dataclass
class InteractionTables:
    """As tabelas produzidas pelo estágio `preprocess` (ver `scripts/pipeline/preprocess.py`)."""

    split_df: pd.DataFrame
    train_table: pd.DataFrame
    val_table: pd.DataFrame
    test_table: pd.DataFrame
    train_eval_table: pd.DataFrame
    all_items: np.ndarray

    @classmethod
    def load(cls, data_dir: Path) -> "InteractionTables":
        """Carrega as tabelas gravadas por `scripts/pipeline/preprocess.py`."""
        interactions_dir = data_dir / "processed" / "interactions"
        return cls(
            split_df=pd.read_parquet(interactions_dir / "split_df.parquet"),
            train_table=pd.read_parquet(interactions_dir / "train_table.parquet"),
            val_table=pd.read_parquet(interactions_dir / "val_table.parquet"),
            test_table=pd.read_parquet(interactions_dir / "test_table.parquet"),
            train_eval_table=pd.read_parquet(
                interactions_dir / "train_eval_table.parquet"
            ),
            all_items=pd.read_parquet(interactions_dir / "all_items.parquet")[
                "itemid"
            ].to_numpy(),
        )

    @property
    def train_events(self) -> pd.DataFrame:
        """Eventos do split `train` (histórico usado para ajustar todas as features)."""
        return self.split_df[self.split_df["split"] == "train"]

    @property
    def raw_tables(self) -> dict[str, pd.DataFrame]:
        """As 4 tabelas de candidatos (sem features), por nome de split."""
        return {
            "train": self.train_table,
            "val": self.val_table,
            "test": self.test_table,
            "train_eval": self.train_eval_table,
        }


@dataclass
class FeaturedTables:
    """Um conjunto de features já persistido em disco (ex.: `fe_v4`, ver `feature_eng.py`)."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_eval: pd.DataFrame
    feature_columns: list[str]
    all_items: np.ndarray

    @classmethod
    def load(cls, data_dir: Path, feature_set_name: str) -> "FeaturedTables":
        """Carrega um conjunto de features gravado por `scripts/pipeline/feature_eng.py`.

        Args:
            data_dir: Diretório base de dados.
            feature_set_name: Nome da pasta em `data/processed/` (ex.: `"fe_v4"`).

        Returns:
            As 4 tabelas, as colunas de features e o universo de itens.
        """
        fe_dir = data_dir / "processed" / feature_set_name
        feature_columns = json.loads((fe_dir / "feature_columns.json").read_text())
        return cls(
            train=pd.read_parquet(fe_dir / "train.parquet"),
            val=pd.read_parquet(fe_dir / "val.parquet"),
            test=pd.read_parquet(fe_dir / "test.parquet"),
            train_eval=pd.read_parquet(fe_dir / "train_eval.parquet"),
            feature_columns=feature_columns,
            all_items=pd.read_parquet(fe_dir / "all_items.parquet")[
                "itemid"
            ].to_numpy(),
        )


def build_fe_v1(tables: InteractionTables) -> dict[str, pd.DataFrame]:
    """fe_v1: atividade do usuário, popularidade do item, recência."""
    basic = BasicFeatures.fit(
        tables.train_events, reference_timestamp=tables.split_df["timestamp"].max()
    )
    return {name: basic.transform(table) for name, table in tables.raw_tables.items()}


def build_fe_v2_variants(
    tables: InteractionTables, raw_events: pd.DataFrame
) -> dict[str, dict[str, pd.DataFrame]]:
    """fe_v2.0-fe_v2.4: as 5 variantes de normalização da co-visitação item-item.

    Args:
        tables: Tabelas do `preprocess`.
        raw_events: Eventos brutos (`events.csv`, com a coluna `event`),
            necessários só para a variante ponderada por evento (`fe_v2.3`).

    Returns:
        Um dict por variante (`"fe_v2.0"`..`"fe_v2.4"`), cada um com as 4
        tabelas (`train`/`val`/`test`/`train_eval`) já com a coluna
        `item_covisitation_score` e as 3 de `fe_v1`.
    """
    basic = BasicFeatures.fit(
        tables.train_events, reference_timestamp=tables.split_df["timestamp"].max()
    )
    covisitation = CovisitationArtifacts.fit(tables.train_events, basic.item_popularity)
    cooc_weighted = covisitation.fit_event_weighted(raw_events, tables.train_events)

    variant_specs = {
        "fe_v2.0": (covisitation.cooc, False),
        "fe_v2.1": (covisitation.cooc, True),
        "fe_v2.2": (covisitation.cooc_cosine, False),
        "fe_v2.3": (cooc_weighted, False),
        "fe_v2.4": (covisitation.cooc_cosine, True),
    }
    result = {}
    for variant_name, (cooc_matrix, normalize) in variant_specs.items():
        scores = compute_covisitation_scores(
            tables.raw_tables, covisitation, cooc_matrix, normalize
        )
        result[variant_name] = {
            name: basic.transform(table.assign(item_covisitation_score=scores[name]))
            for name, table in tables.raw_tables.items()
        }
    return result


def _fit_category_artifacts(
    tables: InteractionTables, data_dir: Path
) -> tuple[CategoryAffinityArtifacts, CategoryAffinityArtifacts, pd.Series]:
    """Ajusta as afinidades de categoria-folha e categoria-pai, e a popularidade relativa."""
    item_to_category = load_item_categories(
        [
            data_dir / "raw" / "item_properties_part1.csv",
            data_dir / "raw" / "item_properties_part2.csv",
        ]
    )
    category_to_parent = load_category_parent_map(
        data_dir / "raw" / "category_tree.csv"
    )
    item_to_parent_category = build_item_to_parent_category(
        item_to_category, category_to_parent
    )

    leaf_affinity = CategoryAffinityArtifacts.fit(tables.train_events, item_to_category)
    parent_affinity = CategoryAffinityArtifacts.fit(
        tables.train_events, item_to_parent_category
    )
    basic = BasicFeatures.fit(
        tables.train_events, reference_timestamp=tables.split_df["timestamp"].max()
    )
    item_relative_popularity = compute_item_relative_popularity(
        basic.item_popularity, leaf_affinity
    )
    return leaf_affinity, parent_affinity, item_relative_popularity


def build_fe_v3(tables: InteractionTables, data_dir: Path) -> dict[str, pd.DataFrame]:
    """fe_v3: fe_v1 + co-visitação cosseno + afinidade de categoria-folha."""
    basic = BasicFeatures.fit(
        tables.train_events, reference_timestamp=tables.split_df["timestamp"].max()
    )
    covisitation = CovisitationArtifacts.fit(tables.train_events, basic.item_popularity)
    covisit_scores = compute_covisitation_scores(
        tables.raw_tables, covisitation, covisitation.cooc_cosine, normalize=False
    )
    leaf_affinity, _, _ = _fit_category_artifacts(tables, data_dir)

    result = {}
    for name, raw_table in tables.raw_tables.items():
        positives_are_train_items = name in ("train", "train_eval")
        featured = raw_table.assign(
            item_covisitation_cosine=covisit_scores[name],
            category_affinity=leaf_affinity.compute_affinity(
                raw_table, positives_are_train_items
            ),
        )
        result[name] = basic.transform(featured)
    return result


def build_fe_v4(tables: InteractionTables, data_dir: Path) -> dict[str, pd.DataFrame]:
    """fe_v4 (produção): fe_v3 + afinidade de categoria-pai + popularidade relativa."""
    basic = BasicFeatures.fit(
        tables.train_events, reference_timestamp=tables.split_df["timestamp"].max()
    )
    covisitation = CovisitationArtifacts.fit(tables.train_events, basic.item_popularity)
    covisit_scores = compute_covisitation_scores(
        tables.raw_tables, covisitation, covisitation.cooc_cosine, normalize=False
    )
    leaf_affinity, parent_affinity, item_relative_popularity = _fit_category_artifacts(
        tables, data_dir
    )

    result = {}
    for name, raw_table in tables.raw_tables.items():
        positives_are_train_items = name in ("train", "train_eval")
        featured = raw_table.assign(
            item_covisitation_cosine=covisit_scores[name],
            category_affinity=leaf_affinity.compute_affinity(
                raw_table, positives_are_train_items
            ),
            parent_category_affinity=parent_affinity.compute_affinity(
                raw_table, positives_are_train_items
            ),
            item_relative_popularity=raw_table["itemid"]
            .map(item_relative_popularity)
            .fillna(0),
        )
        result[name] = basic.transform(featured)
    return result


FeatureSetBuilder = Callable[[InteractionTables, Path], dict[str, pd.DataFrame]]

# Registro dos feature sets elegíveis para produção (persistidos via
# `feature_eng.py`, escolhidos via `scripts/pipeline/promote_feature_set.py`).
# `fe_v2` não entra aqui -- suas 5 variantes existem só para comparação
# (`run_fe_comparison.py`), nenhuma delas é uma versão "final" candidata.
FEATURE_SET_BUILDERS: dict[str, tuple[FeatureSetBuilder, list[str]]] = {
    "fe_v1": (lambda tables, _data_dir: build_fe_v1(tables), FEATURE_COLUMNS_V1),
    "fe_v3": (build_fe_v3, FEATURE_COLUMNS_V3),
    "fe_v4": (build_fe_v4, FEATURE_COLUMNS_V4),
}

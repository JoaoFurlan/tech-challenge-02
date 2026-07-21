"""fe_v3/fe_v4: afinidade usuário-categoria, rollup para categoria-pai e popularidade relativa.

`fe_v3` adiciona `category_affinity` (afinidade usuário-categoria, na
granularidade de categoria-folha). `fe_v4` adiciona `parent_category_affinity`
(mesma lógica, rolada para a categoria-pai via `category_tree.csv` — reduz a
esparsidade de ter milhares de categorias-folha) e `item_relative_popularity`
(popularidade do item relativa à popularidade total da sua própria
categoria-folha).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Self, cast

import numpy as np
import pandas as pd
import scipy.sparse as sp


def load_item_categories(item_properties_paths: list[Path]) -> pd.Series:
    """Lê o `itemid` -> categoria mais recente conhecida, dos arquivos de propriedades.

    A categoria de um item varia no tempo nos dados brutos (múltiplas linhas
    por item); usamos o valor mais recente conhecido por item.

    Args:
        item_properties_paths: Caminhos de `item_properties_part*.csv`.

    Returns:
        Série indexada por `itemid`, com o `categoryid` mais recente.
    """
    category_chunks = []
    for path in item_properties_paths:
        for chunk in pd.read_csv(path, chunksize=500_000):
            category_chunks.append(chunk[chunk["property"] == "categoryid"])
    item_properties_category = pd.concat(category_chunks, ignore_index=True)
    return cast(
        pd.Series,
        item_properties_category.sort_values("timestamp")
        .drop_duplicates("itemid", keep="last")
        .set_index("itemid")["value"]
        .astype(int),
    )


def load_category_parent_map(category_tree_path: Path) -> pd.Series:
    """Lê o mapa `categoryid` -> `parentid` de `category_tree.csv`.

    Args:
        category_tree_path: Caminho de `category_tree.csv`.

    Returns:
        Série indexada por `categoryid`, com o `parentid` correspondente
        (pode conter `NaN` para categorias raiz).
    """
    category_tree = pd.read_csv(category_tree_path)
    return category_tree.set_index("categoryid")["parentid"]


def build_item_to_parent_category(
    item_to_category: pd.Series, category_to_parent: pd.Series
) -> pd.Series:
    """Mapeia cada item à sua categoria-pai (ou à própria, se raiz/sem pai conhecido).

    Args:
        item_to_category: `itemid` -> categoria-folha (`load_item_categories`).
        category_to_parent: Categoria -> categoria-pai (`load_category_parent_map`).

    Returns:
        `itemid` -> categoria-pai.
    """
    return item_to_category.map(category_to_parent).fillna(item_to_category).astype(int)


@dataclass
class CategoryAffinityArtifacts:
    """Matriz usuário-categoria, ajustada sobre os eventos de treino.

    Reutilizável tanto para categoria-folha (`fe_v3`) quanto para
    categoria-pai (`fe_v4`) — só muda qual `item_to_category` é passado a
    `fit`.
    """

    user_to_idx: dict[int, int]
    category_to_idx: dict[int, int]
    item_to_category: pd.Series
    user_category_matrix: sp.csr_matrix
    category_popularity: np.ndarray
    user_total_activity: np.ndarray

    @classmethod
    def fit(cls, train_events: pd.DataFrame, item_to_category: pd.Series) -> Self:
        """Constrói a matriz usuário-categoria a partir dos eventos de treino.

        Args:
            train_events: Eventos do split `train`, com colunas `visitorid`,
                `itemid`.
            item_to_category: `itemid` -> categoria (folha ou pai, conforme a
                granularidade desejada).

        Returns:
            As matrizes ajustadas.
        """
        events_with_category = train_events.assign(
            category=train_events["itemid"].map(item_to_category).fillna(-1).astype(int)
        )
        user_ids = events_with_category["visitorid"].unique()
        category_ids = events_with_category["category"].unique()
        user_to_idx = {u: i for i, u in enumerate(user_ids)}
        category_to_idx = {c: i for i, c in enumerate(category_ids)}

        rows = events_with_category["visitorid"].map(user_to_idx).to_numpy()
        cols = events_with_category["category"].map(category_to_idx).to_numpy()
        data = np.ones(len(events_with_category), dtype=np.float32)
        user_category_matrix = sp.csr_matrix(
            (data, (rows, cols)), shape=(len(user_ids), len(category_ids))
        )
        return cls(
            user_to_idx=user_to_idx,
            category_to_idx=category_to_idx,
            item_to_category=item_to_category,
            user_category_matrix=user_category_matrix,
            category_popularity=np.asarray(user_category_matrix.sum(axis=0)).ravel(),
            user_total_activity=np.asarray(user_category_matrix.sum(axis=1)).ravel(),
        )

    def compute_affinity(
        self, table: pd.DataFrame, positives_are_train_items: bool
    ) -> np.ndarray:
        """Afinidade usuário-categoria, cosseno-normalizada.

        Args:
            table: Tabela de candidatos, com colunas `visitorid`, `itemid` e
                (se `positives_are_train_items`) `label`.
            positives_are_train_items: Se `True`, subtrai 1 da contagem bruta
                para linhas positivas — a própria interação de treino sendo
                pontuada não pode contar como evidência de afinidade para si
                mesma (o mesmo auto-match corrigido na diagonal da
                co-visitação, aqui na forma usuário-categoria).

        Returns:
            Um score de afinidade por linha de `table`.
        """
        user_idx = table["visitorid"].map(self.user_to_idx)
        item_category = (
            table["itemid"].map(self.item_to_category).fillna(-1).astype(int)
        )
        cat_idx = item_category.map(self.category_to_idx)

        valid = (~user_idx.isna()) & (~cat_idx.isna())
        scores = np.zeros(len(table), dtype=np.float32)

        ui = user_idx[valid].to_numpy().astype(np.int64)
        ci = cat_idx[valid].to_numpy().astype(np.int64)
        raw_counts = np.asarray(self.user_category_matrix[ui, ci]).ravel()

        if positives_are_train_items:
            is_positive = table.loc[valid, "label"].to_numpy() == 1
            raw_counts = np.maximum(raw_counts - is_positive.astype(np.float32), 0)

        user_totals = self.user_total_activity[ui]
        cat_pop = self.category_popularity[ci]
        denom = np.sqrt(np.maximum(user_totals, 1) * np.maximum(cat_pop, 1))

        scores[valid.to_numpy()] = raw_counts / denom
        return scores


def compute_item_relative_popularity(
    item_popularity: pd.Series, leaf_category_artifacts: CategoryAffinityArtifacts
) -> pd.Series:
    """Popularidade do item relativa à popularidade total da sua própria categoria-folha.

    Feature só-do-item (sem usuário/label), portanto sem risco de auto-match.

    Args:
        item_popularity: `itemid` -> contagem de interações no treino
            (`features.basic.BasicFeatures.item_popularity`).
        leaf_category_artifacts: Matrizes de afinidade na granularidade de
            categoria-folha (`CategoryAffinityArtifacts.fit` com
            `item_to_category` = categoria-folha, não a de categoria-pai).

    Returns:
        `itemid` -> popularidade relativa à categoria.
    """
    cat_pop_lookup = {
        cat: leaf_category_artifacts.category_popularity[idx]
        for cat, idx in leaf_category_artifacts.category_to_idx.items()
    }
    # pandas-stubs não reconhece Series como Mapping para Index.map(), mas
    # funciona em runtime (o que a notebook original já validou).
    item_categories = (
        item_popularity.index.map(leaf_category_artifacts.item_to_category)  # type: ignore[arg-type]
        .fillna(-1)
        .astype(int)
    )
    category_total_pop = item_categories.map(cat_pop_lookup).fillna(1)
    return pd.Series(
        item_popularity.to_numpy() / np.maximum(category_total_pop.to_numpy(), 1),
        index=item_popularity.index,
        name="item_relative_popularity",
    )

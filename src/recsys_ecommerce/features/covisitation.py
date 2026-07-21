"""fe_v2: co-visitação item-item (quem interage com um item, também interage com quais outros).

Cinco variantes de normalização exploradas no sandbox: soma bruta (`fe_v2.0`),
média por atividade do usuário (`fe_v2.1`), cosseno por popularidade do item
(`fe_v2.2`), ponderada por tipo de evento (`fe_v2.3`), cosseno+média
combinados (`fe_v2.4`). `fe_v3`/`fe_v4` (produção) usam a variante cosseno
(`fe_v2.2`) sem a média adicional.
"""

from dataclasses import dataclass
from typing import Self, cast

import numpy as np
import pandas as pd
import scipy.sparse as sp

EVENT_WEIGHTS = {"view": 1, "addtocart": 3, "transaction": 5}


@dataclass
class CovisitationArtifacts:
    """Matrizes de co-visitação item-item, ajustadas sobre os eventos de treino.

    Attributes:
        item_to_idx: `itemid` -> índice contíguo usado nas matrizes.
        user_train_item_indices: `visitorid` -> lista de índices de item do
            histórico de treino (usado para agregar a co-visitação de um
            candidato com todo o histórico do usuário).
        cooc: Matriz de co-visitação bruta (soma de usuários em comum),
            diagonal zerada.
        cooc_cosine: `cooc` normalizada por popularidade dos dois itens
            (`1/sqrt(pop_i * pop_j)`) — corrige o viés de itens populares
            co-ocorrerem muito só por acaso.
    """

    item_to_idx: dict[int, int]
    user_train_item_indices: dict[int, list[int]]
    cooc: sp.csr_matrix
    cooc_cosine: sp.csr_matrix

    @classmethod
    def fit(cls, train_events: pd.DataFrame, item_popularity: pd.Series) -> Self:
        """Constrói as matrizes de co-visitação a partir dos eventos de treino.

        Args:
            train_events: Eventos do split `train`, com colunas `visitorid`,
                `itemid`.
            item_popularity: `itemid` -> contagem de interações no treino
                (`features.basic.BasicFeatures.item_popularity`), usada na
                normalização cosseno.

        Returns:
            As matrizes ajustadas.
        """
        user_ids = train_events["visitorid"].unique()
        item_ids = train_events["itemid"].unique()
        user_to_idx = {u: i for i, u in enumerate(user_ids)}
        item_to_idx = {it: i for i, it in enumerate(item_ids)}

        user_item_matrix = _build_user_item_matrix(
            train_events, user_to_idx, item_to_idx, weights=None
        )
        cooc = (user_item_matrix.T @ user_item_matrix).tocsr()
        _zero_diagonal(cooc)

        item_popularity_array = item_popularity.reindex(item_ids).fillna(0).to_numpy()
        inv_sqrt_pop = 1.0 / np.sqrt(np.maximum(item_popularity_array, 1))
        pop_scaler = sp.diags(inv_sqrt_pop)
        cooc_cosine = (pop_scaler @ cooc @ pop_scaler).tocsr()

        user_train_item_indices = cast(
            "dict[int, list[int]]",
            train_events.groupby("visitorid")["itemid"]
            .apply(lambda s: [item_to_idx[i] for i in s])
            .to_dict(),
        )
        return cls(
            item_to_idx=item_to_idx,
            user_train_item_indices=user_train_item_indices,
            cooc=cooc,
            cooc_cosine=cooc_cosine,
        )

    def fit_event_weighted(
        self, raw_events: pd.DataFrame, train_events: pd.DataFrame
    ) -> sp.csr_matrix:
        """Constrói a matriz de co-visitação ponderada por tipo de evento (`fe_v2.3`).

        Addtocart/transaction carregam muito mais intenção que um view (funil
        de conversão: view->addtocart ~2.6%, addtocart->transaction ~32%) —
        pondera a matriz usuário-item por isso em vez de tratar todo evento
        como 1. Só usado pelo script de comparação, não pela `fe_v4` de
        produção.

        Args:
            raw_events: Eventos brutos (`events.csv`), com a coluna `event`
                (`view`/`addtocart`/`transaction`).
            train_events: Eventos do split `train` (mesmos usados em `fit`).

        Returns:
            Matriz de co-visitação ponderada, diagonal zerada.
        """
        user_ids = train_events["visitorid"].unique()
        user_to_idx = {u: i for i, u in enumerate(user_ids)}

        train_pairs = train_events[["visitorid", "itemid"]].drop_duplicates()
        train_raw_events = raw_events.merge(
            train_pairs, on=["visitorid", "itemid"], how="inner"
        )
        pair_weights = (
            train_raw_events.assign(weight=train_raw_events["event"].map(EVENT_WEIGHTS))
            .groupby(["visitorid", "itemid"])["weight"]
            .max()
            .reset_index()
        )
        weighted_matrix = _build_user_item_matrix(
            pair_weights, user_to_idx, self.item_to_idx, weights=pair_weights["weight"]
        )
        cooc_weighted = (weighted_matrix.T @ weighted_matrix).tocsr()
        _zero_diagonal(cooc_weighted)
        return cooc_weighted


def _build_user_item_matrix(
    events: pd.DataFrame,
    user_to_idx: dict[int, int],
    item_to_idx: dict[int, int],
    weights: pd.Series | None,
) -> sp.csr_matrix:
    """Monta a matriz esparsa usuário-item (contagem ou peso por par)."""
    rows = events["visitorid"].map(user_to_idx).to_numpy()
    cols = events["itemid"].map(item_to_idx).to_numpy()
    data = (
        np.ones(len(events), dtype=np.float32)
        if weights is None
        else weights.to_numpy(dtype=np.float32)
    )
    return sp.csr_matrix(
        (data, (rows, cols)), shape=(len(user_to_idx), len(item_to_idx))
    )


def _zero_diagonal(matrix: sp.csr_matrix) -> None:
    """Zera a diagonal (auto-covisitação) e libera o espaço, em memória.

    Sem isso, uma linha de treino positiva "encontra a si mesma" no próprio
    histórico do usuário, e `cooc[item, item]` vira só a popularidade do item
    — não um sinal de similaridade de verdade.
    """
    matrix.setdiag(0)
    matrix.eliminate_zeros()


def compute_covisitation_scores(
    tables: dict[str, pd.DataFrame],
    artifacts: CovisitationArtifacts,
    cooc_matrix: sp.csr_matrix,
    normalize: bool,
) -> dict[str, np.ndarray]:
    """Agrega a co-visitação de cada candidato com todo o histórico de treino do usuário.

    Uma única passada por usuário, reaproveitada entre todas as tabelas — o
    custo caro (fatiar+somar linhas de `cooc_matrix`) é por usuário, não por
    tabela.

    Args:
        tables: Tabelas de candidatos por nome (ex.: `{"train": ..., "val": ...}`).
        artifacts: Matrizes/mapas ajustados por `CovisitationArtifacts.fit`.
        cooc_matrix: Qual matriz de co-visitação usar (`cooc`, `cooc_cosine`
            ou a ponderada por evento) — parametrizável para cobrir as 5
            variantes de `fe_v2`.
        normalize: Se `True`, divide a soma pelo tamanho do histórico do
            usuário (`fe_v2.1`/`fe_v2.4`) — torna o score comparável entre
            splits (train tem muitas linhas positivas por usuário, val/test
            só 1, então a soma bruta cresce desproporcionalmente com o
            histórico do usuário).

    Returns:
        Um array de scores por tabela, na mesma ordem/tamanho de `tables[name]`.
    """
    combined = pd.concat(
        [
            df[["visitorid", "itemid"]].assign(__source=name)
            for name, df in tables.items()
        ],
        ignore_index=True,
    )
    scores = np.zeros(len(combined), dtype=np.float32)
    item_idx_arr = combined["itemid"].map(artifacts.item_to_idx).to_numpy()

    for visitorid, group_positions in combined.groupby("visitorid").indices.items():
        hist = artifacts.user_train_item_indices.get(cast(int, visitorid))
        if not hist:
            continue
        positions = np.asarray(group_positions)
        idxs = item_idx_arr[positions]
        valid_mask = ~pd.isna(idxs)
        if not valid_mask.any():
            continue
        valid_positions = positions[valid_mask]
        valid_idxs = idxs[valid_mask].astype(np.int64)
        sub_scores = np.asarray(cooc_matrix[hist, :][:, valid_idxs].sum(axis=0)).ravel()
        if normalize:
            sub_scores = sub_scores / len(hist)
        scores[valid_positions] = sub_scores

    combined["__score"] = scores
    return {
        name: combined.loc[combined["__source"] == name, "__score"].to_numpy()
        for name in tables
    }

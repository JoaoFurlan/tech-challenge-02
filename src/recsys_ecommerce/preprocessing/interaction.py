"""Pré-processamento de interações usuário-item: cold-start, split e negative sampling.

Porta a lógica já validada no sandbox (`new_repo/notebooks/01_eda.ipynb`):
filtro de cold-start convergente, deduplicação, split leave-one-out por
usuário, e negative sampling por rejeição (1:4 no treino, 1:99 na avaliação,
convenção do paper NCF).
"""

import numpy as np
import pandas as pd


def filter_cold_start(df: pd.DataFrame, min_interactions: int) -> pd.DataFrame:
    """Remove usuários/itens com menos de `min_interactions` interações.

    Repete até convergir: remover um usuário pode derrubar um item abaixo do
    limite, e vice-versa.

    Args:
        df: Eventos brutos, com colunas `visitorid`/`itemid`.
        min_interactions: Número mínimo de interações exigido por usuário e
            por item.

    Returns:
        Subconjunto de `df` em que todo usuário e todo item tem pelo menos
        `min_interactions` interações.
    """
    while True:
        user_counts = df.groupby("visitorid").size()
        item_counts = df.groupby("itemid").size()
        valid_users = user_counts[user_counts >= min_interactions].index
        valid_items = item_counts[item_counts >= min_interactions].index
        new_df = df[df["visitorid"].isin(valid_users) & df["itemid"].isin(valid_items)]
        if len(new_df) == len(df):
            return new_df
        df = new_df


def deduplicate_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Remove interações repetidas do mesmo par (usuário, item).

    Mantém a mais recente por timestamp — múltiplos eventos (view,
    addtocart, transaction) do mesmo par viram uma única interação.

    Args:
        df: Eventos com colunas `visitorid`, `itemid`, `timestamp`.

    Returns:
        Um evento por par `(visitorid, itemid)`, o mais recente.
    """
    return df.sort_values("timestamp").drop_duplicates(
        ["visitorid", "itemid"], keep="last"
    )


def assign_loo_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Split leave-one-out por usuário.

    A última interação de cada usuário (por timestamp) vai para `test`, a
    penúltima para `val`, e o restante para `train`.

    Args:
        df: Interações deduplicadas, com colunas `visitorid`, `timestamp`.

    Returns:
        Cópia de `df` ordenada, com uma coluna `split` adicional
        (`"train"`/`"val"`/`"test"`).
    """
    df = df.sort_values(["visitorid", "timestamp"]).copy()
    rank_desc = df.groupby("visitorid").cumcount(ascending=False)
    df["split"] = np.select(
        [rank_desc == 0, rank_desc == 1], ["test", "val"], default="train"
    )
    return df


def sample_negatives(
    user_ids: pd.Series,
    positive_items_by_user: dict[int, set[int]],
    all_items: np.ndarray,
    n_neg_per_user: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Amostra itens negativos (nunca interagidos) por usuário via rejection sampling.

    Sorteia um lote de candidatos, descarta os que colidem com um positivo
    (ou que já foram escolhidos), repete só para o que faltar.

    Args:
        user_ids: Um `visitorid` por linha a amostrar (repetições viram
            amostras independentes, ex.: um `visitorid` presente `k` vezes
            recebe `k` conjuntos de `n_neg_per_user` negativos).
        positive_items_by_user: Mapa `visitorid` -> conjunto de `itemid`
            positivos em QUALQUER split (evita que um negativo amostrado
            seja positivo em outro split).
        all_items: Universo de itens candidatos.
        n_neg_per_user: Quantos negativos amostrar por linha de `user_ids`.
        rng: Gerador de números aleatórios (para reprodutibilidade).

    Returns:
        DataFrame com colunas `visitorid`, `itemid`, `label` (sempre 0),
        `n_neg_per_user` linhas por entrada de `user_ids`.
    """
    all_items = np.asarray(all_items)
    n_items = len(all_items)
    out_users: list[int] = []
    out_items: list[int] = []

    for user_id in user_ids:
        positives = positive_items_by_user[user_id]
        chosen: set[int] = set()
        while len(chosen) < n_neg_per_user:
            missing = n_neg_per_user - len(chosen)
            candidates = all_items[rng.integers(0, n_items, size=missing * 2)]
            for item_id in candidates:
                if item_id not in positives and item_id not in chosen:
                    chosen.add(item_id)
                    if len(chosen) == n_neg_per_user:
                        break
        out_users.extend([user_id] * n_neg_per_user)
        out_items.extend(chosen)

    return pd.DataFrame({"visitorid": out_users, "itemid": out_items, "label": 0})


def build_labeled_table(
    positives: pd.DataFrame, negatives: pd.DataFrame, seed: int
) -> pd.DataFrame:
    """Combina positivos e negativos em uma única tabela embaralhada.

    Args:
        positives: Linhas com `label = 1`.
        negatives: Linhas com `label = 0`.
        seed: Semente do embaralhamento (reprodutibilidade).

    Returns:
        `positives` e `negatives` concatenados e embaralhados.
    """
    table = pd.concat([positives, negatives], ignore_index=True)
    return table.sample(frac=1, random_state=seed).reset_index(drop=True)

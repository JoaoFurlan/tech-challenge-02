"""Métricas de ranking para avaliação leave-one-out (protocolo do paper NCF).

Cada usuário tem exatamente 1 item relevante entre os candidatos avaliados
(o positivo do split) — `evaluate_model` explora essa estrutura fixa para
vetorizar o cálculo via reshape em vez de um laço Python por usuário.
"""

from typing import Any

import numpy as np
import pandas as pd


def evaluate_model(
    model: Any,  # noqa: ANN401
    eval_df: pd.DataFrame,
    feature_columns: list[str],
    all_items: np.ndarray,
    k: int = 10,
) -> dict[str, float]:
    """Avalia um modelo sobre uma tabela de candidatos (protocolo leave-one-out).

    Args:
        model: Qualquer objeto com `predict_proba(X)` estilo sklearn (ex.:
            `recsys_ecommerce.models.base.RecommenderModel`).
        eval_df: Tabela de candidatos com `visitorid`, `itemid`, `label` e as
            colunas em `feature_columns` — grupo de tamanho fixo por usuário
            (1 positivo + N negativos).
        feature_columns: Colunas de features a passar ao modelo.
        all_items: Universo de itens candidatos (para a métrica de cobertura).
        k: Tamanho do ranking considerado nas métricas `@k`.

    Returns:
        Dicionário com `ndcg`, `hit_rate`, `mrr` e `coverage`, cada um a
        média sobre todos os usuários de `eval_df`.
    """
    eval_df = eval_df.sort_values("visitorid")
    scores = model.predict_proba(eval_df[feature_columns])[:, 1]

    n_users = eval_df["visitorid"].nunique()
    assert len(eval_df) % n_users == 0, "grupo com tamanho irregular por usuário"
    group_size = len(eval_df) // n_users

    scores_matrix = scores.reshape(n_users, group_size)
    labels_matrix = eval_df["label"].to_numpy().reshape(n_users, group_size)
    items_matrix = eval_df["itemid"].to_numpy().reshape(n_users, group_size)

    order = np.argsort(-scores_matrix, axis=1)
    ranked_labels = np.take_along_axis(labels_matrix, order, axis=1)[:, :k]
    ranked_items = np.take_along_axis(items_matrix, order, axis=1)[:, :k]

    has_hit = ranked_labels.any(axis=1)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    ndcg = (ranked_labels * discounts).sum(axis=1)
    first_hit_pos = np.argmax(ranked_labels, axis=1)
    mrr = np.where(has_hit, 1.0 / (first_hit_pos + 1), 0.0)
    coverage = len(np.unique(ranked_items)) / len(all_items)

    return {
        "ndcg": float(ndcg.mean()),
        "hit_rate": float(has_hit.mean()),
        "mrr": float(mrr.mean()),
        "coverage": float(coverage),
    }

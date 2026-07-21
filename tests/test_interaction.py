"""Testes para o pré-processamento de interações (cold-start, split, negative sampling)."""

import numpy as np
import pandas as pd

from recsys_ecommerce.preprocessing.interaction import (
    assign_loo_splits,
    build_labeled_table,
    deduplicate_interactions,
    filter_cold_start,
    sample_negatives,
)


def _toy_events() -> pd.DataFrame:
    # 3 usuários, alguns itens com poucas interações (devem cair no cold-start).
    return pd.DataFrame(
        {
            "visitorid": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4],
            "itemid": [10, 11, 12, 10, 11, 13, 10, 11, 12, 99],
            "timestamp": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
        }
    )


def test_filter_cold_start_removes_sparse_users_and_items() -> None:
    events = _toy_events()
    filtered = filter_cold_start(events, min_interactions=3)
    # Usuario 4 (1 interacao) e item 99/13 (1 interacao) devem sumir.
    assert 4 not in filtered["visitorid"].to_numpy()
    assert 99 not in filtered["itemid"].to_numpy()
    assert (filtered.groupby("visitorid").size() >= 3).all()
    assert (filtered.groupby("itemid").size() >= 3).all()


def test_deduplicate_interactions_keeps_latest() -> None:
    df = pd.DataFrame(
        {
            "visitorid": [1, 1],
            "itemid": [10, 10],
            "timestamp": [1, 2],
            "extra": ["old", "new"],
        }
    )
    deduped = deduplicate_interactions(df)
    assert len(deduped) == 1
    assert deduped.iloc[0]["extra"] == "new"


def test_assign_loo_splits_last_two_go_to_test_and_val() -> None:
    df = pd.DataFrame(
        {"visitorid": [1, 1, 1], "itemid": [10, 11, 12], "timestamp": [1, 2, 3]}
    )
    split_df = assign_loo_splits(df)
    assert split_df.loc[split_df["itemid"] == 12, "split"].item() == "test"
    assert split_df.loc[split_df["itemid"] == 11, "split"].item() == "val"
    assert split_df.loc[split_df["itemid"] == 10, "split"].item() == "train"


def test_sample_negatives_never_returns_a_known_positive() -> None:
    positive_items_by_user = {1: {10, 11, 12}}
    all_items = np.array([10, 11, 12, 13, 14, 15])
    rng = np.random.default_rng(0)
    negatives = sample_negatives(
        pd.Series([1]), positive_items_by_user, all_items, n_neg_per_user=3, rng=rng
    )
    assert len(negatives) == 3
    assert (negatives["label"] == 0).all()
    assert not set(negatives["itemid"]).intersection(positive_items_by_user[1])


def test_build_labeled_table_combines_and_shuffles_deterministically() -> None:
    positives = pd.DataFrame({"visitorid": [1], "itemid": [10], "label": 1})
    negatives = pd.DataFrame({"visitorid": [1, 1], "itemid": [11, 12], "label": 0})
    table_a = build_labeled_table(positives, negatives, seed=0)
    table_b = build_labeled_table(positives, negatives, seed=0)
    pd.testing.assert_frame_equal(table_a, table_b)
    assert len(table_a) == 3
    assert set(table_a["label"]) == {0, 1}

"""fe_v1: atividade do usuário, popularidade do item, recência.

Todas calculadas a partir do histórico de TREINO (evita vazamento de val/test
para dentro das features).
"""

from dataclasses import dataclass
from typing import Self

import pandas as pd


@dataclass
class BasicFeatures:
    """Estatísticas simples de usuário/item, ajustadas sobre os eventos de treino.

    Attributes:
        user_activity: `visitorid` -> número de interações no treino.
        item_popularity: `itemid` -> número de interações no treino.
        item_recency_days: `itemid` -> dias desde a última interação
            conhecida (no treino) até `reference_timestamp`.
        max_recency_days: Maior recência observada — usada como valor de
            fallback para itens sem nenhuma interação de treino (o caso mais
            "frio" possível).
    """

    user_activity: pd.Series
    item_popularity: pd.Series
    item_recency_days: pd.Series
    max_recency_days: float

    @classmethod
    def fit(cls, train_events: pd.DataFrame, reference_timestamp: int) -> Self:
        """Calcula as estatísticas a partir dos eventos de treino.

        Args:
            train_events: Eventos do split `train`, com colunas `visitorid`,
                `itemid`, `timestamp`.
            reference_timestamp: Timestamp usado como referência "hoje" para
                calcular a recência (tipicamente o timestamp máximo de todo o
                dataset filtrado).

        Returns:
            As estatísticas ajustadas.
        """
        user_activity = (
            train_events.groupby("visitorid").size().rename("user_activity_count")
        )
        item_popularity = (
            train_events.groupby("itemid").size().rename("item_popularity_count")
        )
        item_last_seen = train_events.groupby("itemid")["timestamp"].max()
        item_recency_days = (
            (reference_timestamp - item_last_seen) / (1000 * 60 * 60 * 24)
        ).rename("item_recency_days")
        return cls(
            user_activity=user_activity,
            item_popularity=item_popularity,
            item_recency_days=item_recency_days,
            max_recency_days=float(item_recency_days.max()),
        )

    def transform(self, table: pd.DataFrame) -> pd.DataFrame:
        """Adiciona as 3 colunas de `fe_v1` a uma tabela de candidatos.

        Usuário/item sem histórico de treino (mesmo sobrevivendo ao filtro
        geral de cold-start, que conta eventos em todos os splits) recebem o
        valor mais "frio" possível: atividade/popularidade zero, recência
        máxima observada.

        Args:
            table: Tabela com colunas `visitorid`, `itemid`.

        Returns:
            Cópia de `table` com as colunas `user_activity_count`,
            `item_popularity_count`, `item_recency_days` adicionadas.
        """
        table = table.merge(
            self.user_activity, left_on="visitorid", right_index=True, how="left"
        )
        table = table.merge(
            self.item_popularity, left_on="itemid", right_index=True, how="left"
        )
        table = table.merge(
            self.item_recency_days, left_on="itemid", right_index=True, how="left"
        )
        table["user_activity_count"] = table["user_activity_count"].fillna(0)
        table["item_popularity_count"] = table["item_popularity_count"].fillna(0)
        table["item_recency_days"] = table["item_recency_days"].fillna(
            self.max_recency_days
        )
        return table


FEATURE_COLUMNS_V1 = [
    "user_activity_count",
    "item_popularity_count",
    "item_recency_days",
]

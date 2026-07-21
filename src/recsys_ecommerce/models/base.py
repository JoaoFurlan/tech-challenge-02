"""Interface de modelos de recomendação, criados via Factory Pattern."""

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import pandas as pd


class RecommenderModel(ABC):
    """Modelo de recomendação treinável sobre uma tabela de features pré-computada.

    Cada subclasse concreta implementa uma arquitetura diferente (árvore,
    regressão logística, MLP), mas todas compartilham o mesmo contrato
    sklearn-like (`fit`/`predict_proba`): a mesma tabela de features
    (`fe_v4`, ver `recsys_ecommerce.features`) alimenta qualquer uma delas, e
    a mesma `evaluate_model` (`recsys_ecommerce.evaluation.metrics`) avalia
    qualquer uma delas sem precisar saber qual é. A criação de instâncias
    concretas é responsabilidade de `ModelFactory`.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """Treina o modelo sobre uma tabela de features e rótulos já prontos.

        Args:
            X: Tabela de features (uma linha por par usuário-item candidato).
            y: Rótulo binário de interação (1 = positivo, 0 = negativo
                amostrado), na mesma ordem de `X`.

        Returns:
            A própria instância, permitindo encadeamento de chamadas.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Prevê a probabilidade de interação para cada linha de `X`.

        Args:
            X: Tabela de features, mesmas colunas usadas em `fit`.

        Returns:
            Array de forma `(len(X), 2)`, estilo sklearn — a coluna 1 é a
            probabilidade da classe positiva (interação).
        """
        raise NotImplementedError

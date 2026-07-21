"""Interface Strategy para etapas de pré-processamento de dados."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class Preprocessor(ABC, Generic[T]):
    """Estratégia de pré-processamento aplicável a um tipo de dado `T`.

    Cada subclasse concreta implementa uma técnica de transformação
    específica (ex.: filtro de cold-start, engenharia de features),
    permitindo trocar a estratégia usada no pipeline sem alterar o código
    que a invoca.
    """

    @abstractmethod
    def fit(self, data: T) -> "Preprocessor[T]":
        """Ajusta o estado interno da estratégia a partir dos dados.

        Args:
            data: Dados de treino usados para calcular os parâmetros da
                transformação (ex.: estatísticas de popularidade/atividade).

        Returns:
            A própria instância, permitindo encadeamento de chamadas.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, data: T) -> T:
        """Aplica a transformação aos dados usando o estado já ajustado.

        Args:
            data: Dados a serem transformados.

        Returns:
            Os dados transformados.
        """
        raise NotImplementedError

    def fit_transform(self, data: T) -> T:
        """Ajusta a estratégia aos dados e em seguida os transforma.

        Args:
            data: Dados de treino, também transformados no retorno.

        Returns:
            Os dados transformados.
        """
        return self.fit(data).transform(data)

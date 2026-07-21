"""Factory Pattern para criação de modelos de recomendação registrados."""

from typing import Any, ClassVar

from recsys_ecommerce.models.base import RecommenderModel


class ModelFactory:
    """Registro central de modelos de recomendação disponíveis.

    Permite que o código cliente crie modelos a partir de um nome (ex.: vindo
    de um arquivo de configuração) sem conhecer a classe concreta, e que
    novos modelos sejam adicionados via `register` sem alterar esta classe.
    """

    _registry: ClassVar[dict[str, type[RecommenderModel]]] = {}

    @classmethod
    def register(cls, name: str, model_cls: type[RecommenderModel]) -> None:
        """Registra uma classe de modelo sob um nome único.

        Args:
            name: Identificador usado para criar o modelo posteriormente.
            model_cls: Classe concreta de `RecommenderModel` a ser registrada.

        Raises:
            ValueError: Se já existir um modelo registrado sob esse nome.
        """
        if name in cls._registry:
            raise ValueError(f"Já existe um modelo registrado com o nome '{name}'.")
        cls._registry[name] = model_cls

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> RecommenderModel:  # noqa: ANN401
        """Cria uma instância do modelo registrado sob o nome informado.

        Args:
            name: Identificador do modelo, definido em uma chamada prévia a
                `register`.
            **kwargs: Argumentos repassados ao construtor do modelo.

        Returns:
            Uma nova instância do modelo solicitado.

        Raises:
            ValueError: Se não houver modelo registrado sob esse nome.
        """
        if name not in cls._registry:
            known = ", ".join(sorted(cls._registry)) or "nenhum"
            raise ValueError(
                f"Modelo '{name}' não registrado. Modelos disponíveis: {known}."
            )
        return cls._registry[name](**kwargs)

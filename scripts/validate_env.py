"""Valida se o ambiente local está pronto para rodar o projeto.

Verifica a versão do Python, a importação das dependências de produção e o
carregamento das configurações a partir do `.env`.

Uso:
    uv run python scripts/validate_env.py
"""

import importlib
import sys
from pathlib import Path

REQUIRED_MODULES = {
    "torch": "PyTorch",
    "sklearn": "Scikit-Learn",
    "mlflow": "MLflow",
    "dvc": "DVC",
    "pandas": "Pandas",
    "pydantic_settings": "Pydantic Settings",
}


def _check_python_version() -> list[str]:
    """Confere se a versão do Python em uso bate com `.python-version`.

    Returns:
        Lista de mensagens de erro (vazia se estiver tudo certo).
    """
    pinned = Path(".python-version").read_text(encoding="utf-8").strip()
    current = f"{sys.version_info.major}.{sys.version_info.minor}"
    if current != pinned:
        return [
            f"Python {pinned} esperado (via .python-version), mas encontrado {current}."
        ]
    return []


def _check_required_modules() -> list[str]:
    """Confere se todas as dependências de produção podem ser importadas.

    Returns:
        Lista de mensagens de erro (vazia se todas importarem corretamente).
    """
    errors = []
    for module_name, display_name in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(f"Falha ao importar {display_name} ({module_name}): {exc}")
    return errors


def _check_settings() -> list[str]:
    """Confere se as configurações do projeto carregam sem erro.

    Returns:
        Lista de mensagens de erro (vazia se as configurações carregarem).
    """
    try:
        from recsys_ecommerce.config import settings

        _ = settings.random_seed
    except Exception as exc:
        return [f"Falha ao carregar Settings: {exc}"]
    return []


def main() -> int:
    """Executa todas as validações e imprime o resultado.

    Returns:
        Código de saída do processo: 0 se tudo passou, 1 caso contrário.
    """
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8")

    checks = [_check_python_version(), _check_required_modules(), _check_settings()]
    errors = [error for check in checks for error in check]

    if errors:
        print("Ambiente INVÁLIDO:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("Ambiente OK: versão do Python, dependências e configurações validadas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

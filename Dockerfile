# syntax=docker/dockerfile:1

# ---- builder: resolve dependências e instala o pacote ----
FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /usr/local/bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Instala as dependências antes de copiar o código-fonte, para que o cache do
# Docker só invalide essa camada quando pyproject.toml/uv.lock mudarem.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-editable --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# ---- runtime: imagem final ----
FROM python:3.11-slim AS runtime
WORKDIR /app

# git é necessário em tempo de execução: o DVC usa o repositório git (mesmo
# sem novos commits) para localizar a raiz do projeto e gerenciar o estado
# dos estágios do pipeline.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /usr/local/bin/
COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY pyproject.toml uv.lock ./
COPY dvc.yaml .dvcignore ./
COPY .dvc ./.dvc
COPY .git ./.git

ENV PATH="/app/.venv/bin:$PATH" \
    UV_NO_SYNC=1

CMD ["sh", "-c", "dvc pull && dvc repro"]

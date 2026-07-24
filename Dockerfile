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

# Este build do PyTorch usa MKL como backend de BLAS (confirmado via
# torch.__config__.show(): BLAS_INFO=mkl, USE_MKL=ON) -- o nn.Linear do MLP
# roda sua multiplicacao de matrizes via MKL, nao via oneDNN/MKL-DNN
# (usado mais para convolucoes/ops fundidas). ONEDNN_MAX_CPU_ISA sozinho
# NAO trava o despacho de instrucoes do MKL -- confirmado: o fix nao mudou
# o resultado entre duas maquinas fisicas diferentes. MKL_CBWR
# ("Conditional Bitwise Reproducibility") e o mecanismo que a própria Intel
# construiu especificamente pra isso: força o MKL a usar sempre o MESMO
# caminho de codigo (aqui, AVX2) em vez de detectar e escolher o mais
# rapido disponivel em cada CPU. Mantém ONEDNN_MAX_CPU_ISA tambem, caso
# alguma operação passe por ali.
ENV PATH="/app/.venv/bin:$PATH" \
    UV_NO_SYNC=1 \
    ONEDNN_MAX_CPU_ISA=AVX2 \
    MKL_CBWR=AVX2

CMD ["sh", "-c", "dvc pull && dvc repro"]

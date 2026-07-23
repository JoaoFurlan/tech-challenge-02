# Sistema de Recomendação de Produtos — E-commerce (resumo do projeto)

> Este arquivo é um resumo prático do projeto: o que é, por que as principais
> decisões foram tomadas, e como executar (nativo ou via Docker). Para o
> detalhamento completo, ver `README.md`, `docs/model-card.md` e
> `docs/decisoes-tecnicas.md`.

## O que é o projeto

Sistema de recomendação de produtos baseado no comportamento de navegação dos
usuários (dataset RetailRocket: visualizações, adições ao carrinho e
transações). Cinco modelos são comparados sob o mesmo protocolo de avaliação:
regressão logística, árvore de decisão, XGBoost, LightGBM e um MLP em
PyTorch.

Stack: **PyTorch** (modelo neural), **Scikit-Learn** (baselines/árvores),
**MLflow** (tracking + Model Registry), **DVC** (versionamento de dados +
pipeline reprodutível), **Docker** (containerização), **uv** (dependências).

## Resultado principal

Com tunagem de hiperparâmetros de verdade (busca real, não só valores
default), o **MLP supera todos os modelos tabulares**:

| Modelo | NDCG@10 | Hit Rate@10 | MRR@10 | Coverage@10 |
|---|---|---|---|---|
| `logreg` | 0,6479 | 0,7975 | 0,5997 | 0,9096 |
| `decision_tree` | 0,4802 | 0,5524 | 0,4574 | 0,9985 |
| `xgboost` | 0,5951 | 0,7215 | 0,5556 | 0,9071 |
| `lightgbm` | 0,6502 | 0,7879 | 0,6069 | 0,9317 |
| `mlp` | **0,6940** | **0,8022** | **0,6593** | 0,9122 |

`mlp` está registrado como `Production` no MLflow Model Registry. Isso não
foi o resultado inicial — a regressão logística venceu por um bom tempo, até
o LightGBM e depois o MLP serem tunados de verdade (ver
`docs/decisoes-tecnicas.md`, decisão 4).

## Por que fizemos o que fizemos (resumo das decisões)

- **MLP com features tabulares, não embeddings puros** — dado o tamanho e a
  esparsidade extrema do dataset, embeddings puros (um vetor por usuário/item)
  generalizariam mal para itens/usuários raros. Features tabulares
  (popularidade, atividade, razões, etc.) compartilhadas com os modelos de
  árvore dão ao MLP o mesmo ponto de partida "justo" para comparação.
- **Feature engineering em camadas (`fe_v1` → `fe_v4`)** — cada versão
  adiciona um grupo de features por cima da anterior, permitindo comparar
  exatamente qual camada ajuda cada modelo, em vez de mudar tudo de uma vez.
- **Dois caminhos separados no pipeline (leve e pesado)** — ver seção
  dedicada abaixo. Existe porque re-treinar do zero (buscas de
  hiperparâmetros reais) leva horas, e isso não deveria ser pré-requisito
  para simplesmente reproduzir os resultados finais.
- **MLflow organizado em runs flat + tags** (não hierarquia aninhada) —
  mais simples de filtrar/comparar na UI do que uma árvore de experimentos
  aninhados; tags (`model_family`, `feature_set`, `trial_type`) fazem o
  trabalho de organização.
- **Promoção automática ao Model Registry por métrica** — o script de
  promoção compara o NDCG@10 de todos os candidatos e promove o vencedor a
  `Production` via alias, sem intervenção manual, para o pipeline ser
  realmente reprodutível ponta a ponta.

## Estrutura do projeto

```
├── src/recsys_ecommerce/   # código-fonte (Factory + Template Method)
├── scripts/pipeline/        # estágios do dvc.yaml + scripts pesados standalone
├── scripts/experiments/     # comparação de feature engineering (pesado)
├── data/raw/                # dados brutos (versionados via DVC)
├── data/processed/          # saídas intermediárias do pipeline (geradas)
├── reports/                 # métricas do pipeline (versionadas via git)
├── configs/                 # hiperparâmetros (model.yaml)
├── notebooks/                # sandbox de exploração (EDA, histórico)
└── docs/                     # Model Card e decisões técnicas
```

## Como executar

Primeiro passo, comum aos dois caminhos abaixo:

```powershell
git clone https://github.com/JoaoFurlan/tech-challenge-02.git
cd tech-challenge-02
```

### Opção A — nativo (sem Docker)

```powershell
uv sync
uv run python scripts/validate_env.py
uv run dvc pull                # baixa data/raw/ do bucket S3 público
```

> **Nota (Windows com usuário acentuado, ex. `João`):** se aparecer
> `No module named 'recsys_ecommerce'` mesmo após `uv sync` limpo, é um bug
> de codificação do `site.py` do Python ao ler o `.pth` do install editável
> (não afeta o Docker). Contorno, uma vez por sessão de terminal:
> ```powershell
> $env:PYTHONPATH = "src"
> ```

Suba um servidor MLflow local (num terminal à parte, deixado rodando):

```powershell
uv run mlflow server --backend-store-uri sqlite:///mlflow-data/mlflow.db --default-artifact-root file:./mlflow-data/artifacts
```

Rode o pipeline (leve, minutos):

```powershell
uv run dvc repro
```

### Opção B — via Docker Compose

```powershell
docker compose up -d --build   # -d = modo detached (não trava o terminal)
docker compose ps               # ver status dos serviços
docker compose logs -f train    # acompanhar o log só do treino
docker compose down              # desligar (mantém o volume do MLflow)
```

O serviço `train` já roda `dvc pull && dvc repro` automaticamente como
comando padrão — puxa `data/raw/` do S3 sozinho (não precisa de `dvc pull`
manual, nem de já ter clonado os dados antes) e então roda o pipeline. Para
re-rodar sem derrubar tudo, **não** passe um comando explícito — isso
substituiria o `dvc pull && dvc repro` padrão e puxaria o tapete da
autossuficiência:

```powershell
docker compose run --rm train
```

Se você tiver certeza de que `data/raw/` já está presente e só quer forçar
o `dvc repro` sem repetir o `dvc pull`, aí sim vale passar o comando
explícito:

```powershell
docker compose run --rm train dvc repro
```

**Importante sobre reprodutibilidade:** rodar nativo (Windows) e via Docker
(Linux) pode produzir números ligeiramente diferentes mesmo com seeds e
`n_jobs` fixos — confirmado com o XGBoost (wheels compiladas diferente por
plataforma). Isso é esperado: reprodutibilidade *entre execuções* (mesmo
ambiente) está garantida; reprodutibilidade *entre ambientes* diferentes
exige o Docker, que é o que padroniza os binários para todo mundo.

### Resetar o pipeline para testar do zero

Antes de rodar de novo do zero (ex.: para validar o caminho Docker depois de
já ter rodado nativo):

```powershell
Remove-Item -Recurse -Force mlflow-data     # se rodou mlflow nativo
Remove-Item -Recurse -Force .dvc\cache
Remove-Item -Recurse -Force data\processed
Remove-Item reports\*.json
Remove-Item dvc.lock
```

`dvc.lock` e `reports/*.json` são versionados no git — apagar localmente é
reversível (`git checkout -- dvc.lock reports/`) a qualquer momento.

## Versão leve vs. pesada

- **Leve (`dvc repro`, obrigatório)** — os estágios do `dvc.yaml`. Cada
  modelo treina só a configuração `tuned` já congelada em
  `configs/model.yaml`. Termina em minutos e já reproduz os números finais
  da tabela acima.
- **Pesada (scripts standalone, opcional)** — a busca de hiperparâmetros de
  verdade que *encontrou* essas configurações vencedoras, e a comparação
  completa de feature engineering (`fe_v1`..`fe_v4`). Fica fora do
  `dvc.yaml` de propósito — pode levar horas e só é necessária se quiser
  tentar superar os hiperparâmetros já congelados.

```powershell
uv run python scripts/pipeline/run_heavy_exploration.py   # tudo de uma vez, resumível
```

Não tem um serviço/comando dedicado no `docker-compose.yml` pra isso — só o
caminho leve (`dvc pull && dvc repro`) roda por padrão. Mas a imagem já tem
tudo que a busca pesada precisa (código, dependências, `MLFLOW_TRACKING_URI`
já apontado pro serviço `mlflow`), então dá pra rodar dentro do container
sobrescrevendo o comando:

```powershell
docker compose run --rm train uv run python scripts/pipeline/run_heavy_exploration.py
```

## Testes e MLflow UI

```powershell
uv run pytest
uv run ruff check .
uv run mypy
```

MLflow UI: `http://localhost:5000` (com o servidor rodando, nativo ou via
Docker).

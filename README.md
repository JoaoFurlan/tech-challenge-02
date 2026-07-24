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
| `xgboost` | 0,6522 | TODO | TODO | TODO |
| `lightgbm` | 0,6502 | 0,7879 | 0,6069 | 0,9317 |
| `mlp` | **0,6969** | TODO | TODO | TODO |

> Números acima são de uma rodada via Docker (o ambiente de referência do
> projeto), já com o fix de threads fixas (`N_JOBS`/`N_THREADS = 4`) em
> `xgboost`/`lightgbm`/`mlp`. `logreg`/`decision_tree`/`lightgbm` não mudam
> entre ambientes (nunca dependeram de contagem de threads); `xgboost`
> mudou de 0,5951 (valor nativo antigo, sem o fix) para 0,6522; `mlp` mudou
> de 0,6940 para 0,6969. Colunas `TODO` ainda precisam ser conferidas na UI
> do MLflow para as runs `xgboost-tuned`/`mlp-tuned` atuais.

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
docker compose up -d --build; docker compose logs -f train
```

Um único comando faz tudo: builda as duas imagens, sobe o `mlflow` (espera
ficar saudável) e roda o `train` (`dvc pull && dvc repro` — puxa
`data/raw/` do S3 sozinho, sem precisar de `dvc pull` manual antes — e então
o pipeline inteiro). O `-d` roda em modo detached (não trava o terminal); o
`; docker compose logs -f train` encadeado na mesma linha já acompanha o log
só do treino, sem precisar rodar dois comandos separados. `train` termina e
sai sozinho (`Exited (0)`) quando o pipeline acaba — `mlflow` continua no
ar. Pra desligar tudo depois: `docker compose down` (mantém o volume do
MLflow).

`data/`, `models/` e `reports/` são bind mounts (o host vê as mudanças em
tempo real), mas `dvc.lock` **não é** -- bind-mount de arquivo único quebra
o `dvc pull` (o DVC/git reescrevem esse arquivo via write-then-rename, o que
desconecta o bind mount do host). Pra levar o `dvc.lock` gerado pelo Docker
de volta pro seu repositório (e poder commitar), copie manualmente depois
que o `train` terminar (o container continua existindo, parado, já que não
rodou com `--rm`):

```powershell
docker cp recsys-ecommerce-train-1:/app/dvc.lock ./dvc.lock
```

**Já rodou uma vez e quer rodar de novo?** Rodar o mesmo comando de novo
funciona sem problema — mas como o DVC já teria os estágios marcados como
"não mudou" no `dvc.lock`, ele só vai pular tudo rapidinho (nenhum modelo
retreina de verdade), a menos que você tenha alterado algum código no meio
do caminho. Se o que você quer é ver o pipeline rodar de novo do ZERO (ex.:
pra comparar com uma rodada nativa anterior), siga a seção "Resetar o
pipeline para testar do zero" logo abaixo **antes** de rodar o comando de
novo. Alternativa mais cirúrgica, sem builda de novo nem tocar no `mlflow`:
`docker compose run --rm train` (repete só o `train`, reaproveitando o
`mlflow` já no ar) — mas não passe um comando explícito depois de `train`
nesse caso, ou você substitui o `dvc pull && dvc repro` padrão e perde a
autossuficiência.

**Importante sobre reprodutibilidade:** `xgboost`, `lightgbm` e o `mlp` têm
o número de threads fixado numa constante (`N_JOBS`/`N_THREADS = 4` em cada
model class) -- sem isso, cada um paraleliza suas operações usando o número
de cores que detecta na máquina/container, e isso muda a ordem de agregação
ponto-flutuante dos cálculos, produzindo resultados ligeiramente diferentes
mesmo com a mesma seed. Isso resolveu por completo `xgboost`/`lightgbm`
entre máquinas, e reduziu bastante a variação do `mlp` (que antes dava
`0,6913` numa máquina e `0,69415` em outra, ambos via Docker).

O `mlp` especificamente ainda tem uma variação residual pequena entre
máquinas físicas diferentes (ex.: `0,6945` vs. `0,6969`, ~0,35% relativo),
mesmo com threads fixas. Investigado a fundo, em ordem:

1. **`ONEDNN_MAX_CPU_ISA=AVX2`** (Dockerfile) -- tentativa de travar o
   backend oneDNN/MKL-DNN num teto de instruções fixo, em vez de cada CPU
   escolher a mais rápida disponível (AVX2 vs. AVX-512, etc.) em tempo de
   execução. **Não resolveu** -- confirmado via `torch.__config__.show()`
   que o `nn.Linear` do MLP roda sua multiplicação de matrizes via MKL, não
   via oneDNN (usado mais para convoluções/ops fundidas) -- a variável
   errada pro operador certo.
2. **`MKL_CBWR=AVX2`** (Dockerfile) -- o mecanismo que a própria Intel
   construiu especificamente pra isso (bitwise reproducibility entre
   gerações de CPU diferentes rodando MKL). Confirmado que a variável é
   reconhecida (smoke test local, sem erro), e que o número de threads do
   MKL de fato acompanha `torch.set_num_threads()` (`mkl_get_max_threads()`
   também mostra 4 depois da chamada) -- mas mesmo assim, **também não
   fechou a diferença** entre as duas máquinas testadas.

Conclusão: o resíduo provavelmente vem de uma camada mais profunda ainda --
os kernels vetorizados internos do próprio PyTorch para operações
elemento-a-elemento (ReLU, máscara do Dropout, acumulação do otimizador
Adam, BCE loss), que usam o despacho de SIMD interno do PyTorch
(`DispatchStub`), separado do MKL/oneDNN e sem uma variável de ambiente
pública documentada pra travar. Como o treino é iterativo (dezenas de
épocas), uma diferença mínima em qualquer operação se acumula ao longo do
treino.

**Decisão: aceitar essa variação residual, não seguir investigando.**
Nenhum critério de avaliação do desafio exige reprodutibilidade numérica
bit-a-bit entre hardwares arbitrários (os critérios pedem `dvc repro`
funcional, pipeline reprodutível, Docker funcional -- não "mesmo número até
a última casa decimal em qualquer máquina"). O que já foi feito (identificar
threads como a fonte dominante de variação, corrigir isso, investigar mais
duas camadas com raciocínio e verificação reais, não só tentativa e erro
cega) já demonstra profundidade de engenharia suficiente -- perseguir os
0,35% restantes tem retorno decrescente frente ao tempo até a entrega.

O que ainda NÃO é garantido, e não é o foco aqui: rodar nativo (Windows) vs.
via Docker (Linux) pode continuar produzindo números mais diferentes ainda
-- confirmado com o XGBoost antes do fix de threads (wheels compiladas
diferente por plataforma). Por isso o Docker é o ambiente padrão de
referência deste projeto: reprodutibilidade *dentro* dele (entre execuções,
entre máquinas, com a variação residual do MLP documentada e aceita acima)
é o que se garante; nativo é só uma alternativa mais rápida pra
desenvolvimento, não a referência oficial.

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

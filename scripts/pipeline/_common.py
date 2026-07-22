"""Funções compartilhadas pelos scripts de treino de modelos tabulares (`train_*.py`).

Não é um estágio do `dvc.yaml` em si — helper reaproveitado por
`train_baseline.py`/`train_decision_tree.py`/`train_xgboost.py`/`train_lightgbm.py`.
Cada um roda 1 trial flat (`tuned`, os vencedores de `configs/model.yaml`)
sobre o `feature_set` vencedor (`configs/model.yaml:winning_feature_set`,
decisão manual trocada via `scripts/pipeline/promote_feature_set.py`), e
registra o resultado no Model Registry -- sem categorias/nesting, sem
retreino separado.

Não roda mais um trial `baseline` (hiperparâmetros default) -- em todos os
casos observados o `tuned` já vencia, então o baseline só consumia tempo de
treino sem mudar a decisão final. Ver decisão correspondente em
`docs/decisoes-tecnicas.md`.
"""

import json
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow

from recsys_ecommerce.config import load_training_config, settings
from recsys_ecommerce.evaluation.metrics import evaluate_model
from recsys_ecommerce.evaluation.plots import (
    plot_feature_importance,
    plot_metrics_across_splits,
)
from recsys_ecommerce.features.pipeline import FeaturedTables
from recsys_ecommerce.models.base import RecommenderModel
from recsys_ecommerce.tracking.mlflow_organization import (
    find_run_id,
    log_model_artifact,
    register_best_trial,
)

EXPERIMENT_NAME = "model-selection"
REPORTS_DIR = Path("reports")

# MLflow imprime um emoji (URL da run ao encerrar) no stdout -- o console do
# Windows usa cp1252 por padrao, que nao consegue codificar isso e quebra o
# script. Mesma correcao ja usada em scripts/validate_env.py.
if (reconfigure := getattr(sys.stdout, "reconfigure", None)) is not None:
    reconfigure(encoding="utf-8")

# Silencia o ruido cosmetico que o MLflow imprime a cada log_model() --
# resolucao de dependencias via uv export, aviso de indice privado, aviso de
# seguranca do formato pickle (ja e uma escolha deliberada pro MLP, ver
# NeuralMLPModel) etc. Nao muda nenhum comportamento, so o nivel de log;
# erros de verdade (logging.ERROR ou acima) continuam aparecendo.
logging.getLogger("mlflow").setLevel(logging.ERROR)


@contextmanager
def log_stage_timing(stage_name: str) -> Iterator[None]:
    """Imprime data/hora de início e fim (+ duração) de um trecho do script.

    Não é um substituto do timestamp que o MLflow já grava em cada run --
    serve pra dar visibilidade no console de quanto tempo cada estágio (ou,
    na busca pesada, cada família de modelo) está levando, sem precisar
    estimar pelos horários espalhados no log como foi feito manualmente
    numa rodada anterior.

    Args:
        stage_name: Nome a exibir (ex.: `"train_lightgbm"`, `"xgboost-fe_v4"`).
    """
    start = datetime.now()
    print(f"[{start:%Y-%m-%d %H:%M:%S}] iniciando {stage_name}")
    t0 = time.monotonic()
    try:
        yield
    finally:
        elapsed = timedelta(seconds=round(time.monotonic() - t0))
        end = datetime.now()
        print(f"[{end:%Y-%m-%d %H:%M:%S}] {stage_name} concluído em {elapsed}")


def load_winning_feature_set() -> str:
    """Lê o `feature_set` vencedor de `configs/model.yaml:winning_feature_set`.

    Decisão manual (revisada em `scripts/experiments/run_fe_comparison.py`,
    trocada via `scripts/pipeline/promote_feature_set.py`) -- não mais lida
    de `reports/feature_engineering_winner.json`, que hoje é só um relatório
    informativo.

    Returns:
        O nome do feature set vencedor (ex.: `"fe_v4"`).
    """
    return load_training_config().winning_feature_set


def load_featured_tables(feature_set: str) -> FeaturedTables:
    """Carrega as tabelas de `feature_set`, gerando os dados a partir do zero se preciso.

    Usado pelos scripts de tunagem pesada (`tune_tabular_models.py`,
    `tune_neural_mlp.py`) para não depender de `dvc repro` já ter rodado --
    um clone novo que só quer rodar a busca pesada, sem nunca ter passado
    pelo pipeline leve, ainda funciona: roda `preprocess` e/ou `feature_eng`
    primeiro, só se os dados ainda não existirem em disco.

    Args:
        feature_set: Nome do feature set a carregar (ex.: `"fe_v4"`).

    Returns:
        As 4 tabelas, as colunas de features e o universo de itens.
    """
    from feature_eng import run_feature_eng
    from preprocess import run_preprocess

    interactions_dir = settings.data_dir / "processed" / "interactions"
    if not interactions_dir.exists():
        print(f"'{interactions_dir}' não existe -- rodando preprocess.py primeiro.")
        cfg = load_training_config()
        run_preprocess(
            data_dir=settings.data_dir,
            min_interactions=cfg.min_interactions,
            negative_sampling_ratio=cfg.negative_sampling_ratio,
            eval_negative_samples=cfg.eval_negative_samples,
            seed=settings.random_seed,
        )

    feature_dir = settings.data_dir / "processed" / feature_set
    if not feature_dir.exists():
        print(f"'{feature_dir}' não existe -- rodando feature_eng.py primeiro.")
        run_feature_eng(settings.data_dir, feature_set)

    return FeaturedTables.load(settings.data_dir, feature_set)


def train_and_log_trial(
    model_family: str,
    trial_type: str,
    model: RecommenderModel,
    feature_set: str,
    tables: FeaturedTables,
) -> tuple[str, float]:
    """Loga (ou retoma) um trial flat: run `{model_family}-{trial_type}`.

    Idempotente: pula o treino se a run já existir, relendo `test_ndcg` do
    que já está logado.

    Args:
        model_family: Tag `model_family` (ex.: `"logreg"`).
        trial_type: Tag `trial_type` (`"baseline"` ou `"tuned"`).
        model: Instância do modelo (ainda não treinada).
        feature_set: Tag `feature_set` (ex.: `"fe_v4"`).
        tables: Tabelas de features já carregadas.

    Returns:
        `(run_id, test_ndcg)`.
    """
    run_name = f"{model_family}-{trial_type}"
    existing_id = find_run_id(EXPERIMENT_NAME, run_name)
    if existing_id is not None:
        print(f"{run_name} já logado, pulando.")
        run = mlflow.MlflowClient().get_run(existing_id)
        return existing_id, run.data.metrics["test_ndcg"]

    X_train = tables.train[tables.feature_columns]
    y_train = tables.train["label"]
    model.fit(X_train, y_train)

    train_metrics = evaluate_model(
        model, tables.train_eval, tables.feature_columns, tables.all_items
    )
    val_metrics = evaluate_model(
        model, tables.val, tables.feature_columns, tables.all_items
    )
    test_metrics = evaluate_model(
        model, tables.test, tables.feature_columns, tables.all_items
    )

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "model_family": model_family,
                "feature_set": feature_set,
                "trial_type": trial_type,
            }
        )
        get_params = getattr(model, "get_params", None)
        if get_params is not None:
            mlflow.log_params(get_params())
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        fig_splits = plot_metrics_across_splits(
            train_metrics, val_metrics, test_metrics, run_name
        )
        mlflow.log_figure(fig_splits, "metrics_by_split.png")
        plt.close(fig_splits)

        underlying = getattr(model, "underlying_estimator", model)
        fig_importance = plot_feature_importance(
            underlying, tables.feature_columns, model_family
        )
        mlflow.log_figure(fig_importance, "feature_importance.png")
        plt.close(fig_importance)

        log_model_artifact(model)

    print(f"{run_name} -- test_ndcg={test_metrics['ndcg']:.4f}")
    return run.info.run_id, test_metrics["ndcg"]


def run_tuned(model_family: str, tuned_model: RecommenderModel) -> None:
    """Roda o trial `tuned` de uma família e registra o candidato no Model Registry.

    Args:
        model_family: Nome da família de modelo (ex.: `"logreg"`).
        tuned_model: Instância com os hiperparâmetros vencedores (`configs/model.yaml`).
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with log_stage_timing(f"train_{model_family}"):
        feature_set = load_winning_feature_set()
        tables = FeaturedTables.load(settings.data_dir, feature_set)

        _, tuned_ndcg = train_and_log_trial(
            model_family, "tuned", tuned_model, feature_set, tables
        )

        cfg = load_training_config()
        register_best_trial(EXPERIMENT_NAME, model_family, cfg.registered_model_name)

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / f"{model_family}_metrics.json").write_text(
            json.dumps({"test_ndcg": tuned_ndcg}, indent=2)
        )

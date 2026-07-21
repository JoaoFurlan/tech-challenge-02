"""Organização de runs no MLflow: runs flat (tags), busca de hiperparâmetros e registro.

Todas as runs de um experimento ficam no mesmo nível (sem nesting) —
`model_family`/`feature_set`/`trial_type` (tags) são o que se filtra/ordena
na UI para achar o resultado que importa, em vez de precisar abrir uma
árvore de runs pai/filho. `register_best_trial` varre os trials já logados
de uma família e registra o de melhor `test_ndcg` no Model Registry, sem
precisar retreinar nada à parte.
"""

from collections.abc import Callable
from typing import Any, cast

import mlflow
import mlflow.lightgbm
import mlflow.pytorch
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from sklearn.model_selection import ParameterSampler

from recsys_ecommerce.evaluation.metrics import evaluate_model
from recsys_ecommerce.models.lightgbm_model import LightGBMModel
from recsys_ecommerce.models.neural_mlp import NeuralMLPModel
from recsys_ecommerce.models.xgboost_model import XGBoostModel


def run_exists(experiment_name: str, run_name: str) -> bool:
    """Confere se já existe uma run com este nome no experimento.

    Usado para pular trabalho já feito em uma reexecução (idempotência).

    Args:
        experiment_name: Nome do experimento MLflow.
        run_name: Nome exato da run (`tags.mlflow.runName`).

    Returns:
        `True` se uma run com esse nome já existe.
    """
    return find_run_id(experiment_name, run_name) is not None


def find_run_id(
    experiment_name: str, run_name: str, tuning_target: str | None = None
) -> str | None:
    """Busca o `run_id` de uma run pelo nome, opcionalmente escopado por `tuning_target`.

    O escopo por `tuning_target` é necessário para nomes de trial
    (`trial-000`, `trial-001`...), que se repetem entre buscas diferentes —
    sem ele, retomar a busca B encontraria por engano o `trial-000` da busca A.

    Args:
        experiment_name: Nome do experimento MLflow.
        run_name: Nome exato da run (`tags.mlflow.runName`).
        tuning_target: Se informado, exige também `tags.tuning_target` igual
            a este valor.

    Returns:
        O `run_id` da run mais recente com esse nome (e escopo), ou `None`
        se nenhuma existir.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    filter_string = f"tags.mlflow.runName = '{run_name}'"
    if tuning_target is not None:
        filter_string += f" and tags.tuning_target = '{tuning_target}'"
    existing = cast(
        pd.DataFrame,
        mlflow.search_runs(
            experiment_ids=[experiment.experiment_id], filter_string=filter_string
        ),
    )
    if existing.empty:
        return None
    return str(existing.iloc[0]["run_id"])


def log_model_artifact(model: Any, name: str = "model") -> None:  # noqa: ANN401
    """Loga o artefato de um modelo treinado, no flavor MLflow nativo apropriado.

    XGBoost/LightGBM têm tipos internos (ex.: `xgboost.core.Booster`) que o
    serializador do flavor `sklearn` (via `skops`) rejeita por padrão como
    "não confiável" — os flavors nativos evitam esse problema. O MLP usa
    `serialization_format="pickle"` em vez do default (`pt2`, via
    `torch.export`) porque o `torch.export` marca a dimensão de batch como
    dinâmica por padrão, o que conflita com um `input_example` de batch
    size 1 (`ConstraintViolationError`).
    """
    if isinstance(model, NeuralMLPModel):
        mlflow.pytorch.log_model(
            model.underlying_estimator.model_, name=name, serialization_format="pickle"
        )
    elif isinstance(model, XGBoostModel):
        mlflow.xgboost.log_model(model.underlying_estimator, name=name)
    elif isinstance(model, LightGBMModel):
        mlflow.lightgbm.log_model(model.underlying_estimator, name=name)
    else:
        mlflow.sklearn.log_model(model.underlying_estimator, name=name)


def register_best_trial(
    experiment_name: str, model_family: str, registered_model_name: str
) -> str | None:
    """Registra, no Model Registry, o trial de melhor `test_ndcg` já logado para esta família.

    Consulta o MLflow em vez de receber os candidatos do chamador -- assim
    funciona mesmo quando os trials de uma família vêm de scripts
    diferentes (ex.: `train_neural_mlp.py` e `tune_neural_mlp.py`, ambos
    contribuindo runs com `model_family=mlp`). Idempotente por `run_id`: se
    o run vencedor já tem uma versão registrada apontando pra ele, não
    duplica.

    Args:
        experiment_name: Nome do experimento MLflow (`model-selection`).
        model_family: Valor da tag `model_family` a comparar (ex.: `"mlp"`).
        registered_model_name: Nome do modelo no Model Registry.

    Returns:
        A versão registrada (nova ou já existente), ou `None` se nenhum
        trial desta família foi encontrado.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    runs = cast(
        pd.DataFrame,
        mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.model_family = '{model_family}'",
        ),
    )
    runs = runs[runs["metrics.test_ndcg"].notna()]
    if runs.empty:
        return None
    best = cast(pd.Series, runs.loc[runs["metrics.test_ndcg"].idxmax()])
    best_run_id, best_ndcg = str(best["run_id"]), float(best["metrics.test_ndcg"])

    client = mlflow.MlflowClient()
    existing = client.search_model_versions(f"run_id='{best_run_id}'")
    if existing:
        print(
            f"[{model_family}] melhor candidato já registrado: versão {existing[0].version}"
        )
        return existing[0].version

    model_uri = f"runs:/{best_run_id}/model"
    registered = mlflow.register_model(model_uri, registered_model_name)
    client.set_model_version_tag(
        registered_model_name, registered.version, "model_family", model_family
    )
    client.set_model_version_tag(
        registered_model_name, registered.version, "test_ndcg", f"{best_ndcg:.6f}"
    )
    client.update_model_version(
        name=registered_model_name,
        version=registered.version,
        description=(
            f"Melhor trial de `{model_family}` (run `{best_run_id}`, test_ndcg={best_ndcg:.4f})."
        ),
    )
    print(
        f"[{model_family}] registrado: versão {registered.version} (test_ndcg={best_ndcg:.4f})"
    )
    return registered.version


def run_hyperparameter_search(
    search_name: str,
    trial_type: str,
    model_class: Callable[..., Any],
    search_space: dict[str, list[Any]],
    n_trials: int,
    model_family: str,
    feature_set: str,
    experiment_name: str,
    train_feat: pd.DataFrame,
    val_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    train_eval_feat: pd.DataFrame,
    feature_columns: list[str],
    all_items: Any,  # noqa: ANN401
    fixed_params: dict[str, Any] | None = None,
    seed: int = 42,
) -> list[tuple[str, float]]:
    """Busca aleatória de hiperparâmetros: runs flat, uma por trial, artefato incluso.

    Cada trial já loga o próprio artefato (via `log_model_artifact`) -- não
    há retreino separado para "o melhor"; quem vence só precisa ser
    registrado (ver `register_best_trial`). Resumível de verdade: se
    interrompido no meio, uma nova chamada pula os trials já logados
    individualmente (escopados por `search_name` via `tuning_target`) e
    continua do ponto certo.

    Args:
        search_name: Identificador desta busca (ex.: `"mlp-fe_v4-tuned"`) --
            cada trial se chama `{search_name}-trial-000` etc.
        trial_type: Tag `trial_type` a aplicar em cada trial (ex.:
            `"baseline"`/`"tuned"`).
        model_class: Construtor que aceita os hiperparâmetros amostrados
            como kwargs e retorna um `RecommenderModel` (`.fit`/`.predict_proba`).
        search_space: Espaço de busca no formato do `ParameterSampler` do
            scikit-learn.
        n_trials: Número de combinações a amostrar.
        model_family: Tag `model_family` (ex.: `"mlp"`).
        feature_set: Tag `feature_set` (ex.: `"fe_v4"`).
        experiment_name: Nome do experimento MLflow ativo.
        train_feat: Tabela de treino (features + `label`).
        val_feat: Tabela de validação.
        test_feat: Tabela de teste.
        train_eval_feat: Tabela de diagnóstico de treino (mesmo formato de
            val/test, ver `scripts/pipeline/preprocess.py`).
        feature_columns: Colunas de features a usar.
        all_items: Universo de itens candidatos (para a métrica de cobertura).
        fixed_params: Hiperparâmetros fixos, iguais em todos os trials (ex.:
            `max_epochs`, `random_state`).
        seed: Semente do `ParameterSampler`.

    Returns:
        Lista `(run_id, test_ndcg)` de cada trial (novo ou já existente).
    """
    fixed_params = fixed_params or {}
    param_list = list(
        ParameterSampler(search_space, n_iter=n_trials, random_state=seed)
    )
    X_train, y_train = train_feat[feature_columns], train_feat["label"]
    client = mlflow.MlflowClient()

    results: list[tuple[str, float]] = []
    val_ndcgs: list[float] = []

    for i, params in enumerate(param_list):
        trial_name = f"{search_name}-trial-{i:03d}"
        existing_id = find_run_id(
            experiment_name, trial_name, tuning_target=search_name
        )
        if existing_id is not None:
            existing_run = client.get_run(existing_id)
            results.append((existing_id, existing_run.data.metrics["test_ndcg"]))
            val_ndcgs.append(existing_run.data.metrics["val_ndcg"])
            print(f"  [{search_name}] {trial_name} já logado, pulando.")
            continue

        full_params = {**fixed_params, **params}
        model = model_class(**full_params)
        model.fit(X_train, y_train)

        train_metrics = evaluate_model(
            model, train_eval_feat, feature_columns, all_items
        )
        val_metrics = evaluate_model(model, val_feat, feature_columns, all_items)
        test_metrics = evaluate_model(model, test_feat, feature_columns, all_items)

        with mlflow.start_run(run_name=trial_name) as trial_run:
            mlflow.set_tags(
                {
                    "model_family": model_family,
                    "feature_set": feature_set,
                    "trial_type": trial_type,
                    "tuning_target": search_name,
                }
            )
            mlflow.log_params(full_params)
            mlflow.log_metrics({f"train_{m}": v for m, v in train_metrics.items()})
            mlflow.log_metrics({f"val_{m}": v for m, v in val_metrics.items()})
            mlflow.log_metrics({f"test_{m}": v for m, v in test_metrics.items()})
            log_model_artifact(model)

        results.append((trial_run.info.run_id, test_metrics["ndcg"]))
        val_ndcgs.append(val_metrics["ndcg"])
        print(
            f"  [{search_name}] {trial_name}: "
            f"val_ndcg={val_metrics['ndcg']:.4f}, test_ndcg={test_metrics['ndcg']:.4f}"
        )

    best_i = int(pd.Series(val_ndcgs).idxmax())
    print(f"[{search_name}] melhor trial (por val_ndcg): trial-{best_i:03d}")
    return results

"""Testes para a organização flat de runs no MLflow (tags, busca, registro)."""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from recsys_ecommerce.tracking.mlflow_organization import (
    find_run_id,
    register_best_trial,
    run_exists,
    run_hyperparameter_search,
)

EXPERIMENT_NAME = "test-experiment"


@pytest.fixture(autouse=True)
def _isolated_tracking_uri(tmp_path: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)


def test_run_exists_false_then_true() -> None:
    assert run_exists(EXPERIMENT_NAME, "minha-run") is False
    with mlflow.start_run(run_name="minha-run"):
        pass
    assert run_exists(EXPERIMENT_NAME, "minha-run") is True


def test_find_run_id_scopes_by_tuning_target() -> None:
    with mlflow.start_run(run_name="trial-000"):
        mlflow.set_tag("tuning_target", "search-a")
    with mlflow.start_run(run_name="trial-000"):
        mlflow.set_tag("tuning_target", "search-b")

    id_a = find_run_id(EXPERIMENT_NAME, "trial-000", tuning_target="search-a")
    id_b = find_run_id(EXPERIMENT_NAME, "trial-000", tuning_target="search-b")

    assert id_a is not None
    assert id_b is not None
    assert id_a != id_b


class _FitCountingModel:
    """Modelo fake só para contar quantas vezes `fit` é chamado (sem treinar nada de verdade)."""

    n_fit_calls = 0

    def __init__(self, multiplier: float = 1.0) -> None:
        self.multiplier = multiplier

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_FitCountingModel":
        _FitCountingModel.n_fit_calls += 1
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        scores = (X["f0"].to_numpy() * self.multiplier + 1) / 2
        scores = np.clip(scores, 0.01, 0.99)
        return np.column_stack([1 - scores, scores])

    def get_params(self) -> dict[str, object]:
        return {"multiplier": self.multiplier}

    @property
    def underlying_estimator(self) -> "_FitCountingModel":
        return self


def _toy_eval_table(
    n_users: int = 5, group_size: int = 10, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = []
    for user in range(n_users):
        items = rng.permutation(100)[:group_size]
        labels = [1] + [0] * (group_size - 1)
        for item, label in zip(items, labels, strict=True):
            rows.append(
                {"visitorid": user, "itemid": item, "label": label, "f0": rng.normal()}
            )
    return pd.DataFrame(rows)


def _fake_log_model_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_FitCountingModel` não é um `sklearn`/`xgboost`/`lightgbm`/MLP de verdade --
    troca `log_model_artifact` por um no-op pra não quebrar o dispatch de flavor."""
    monkeypatch.setattr(
        "recsys_ecommerce.tracking.mlflow_organization.log_model_artifact",
        lambda model: None,
    )


def test_run_hyperparameter_search_resumes_without_recomputing_finished_trials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_log_model_artifact(monkeypatch)
    _FitCountingModel.n_fit_calls = 0
    train_feat = _toy_eval_table(seed=1)
    val_feat = _toy_eval_table(seed=2)
    test_feat = _toy_eval_table(seed=3)
    train_eval_feat = _toy_eval_table(seed=4)
    all_items = np.arange(100)

    def _run() -> list[tuple[str, float]]:
        return run_hyperparameter_search(
            search_name="fake-model",
            trial_type="tuned",
            model_class=_FitCountingModel,
            search_space={"multiplier": [1.0, 2.0, 3.0]},
            n_trials=3,
            model_family="fake",
            feature_set="fe_v4",
            experiment_name=EXPERIMENT_NAME,
            train_feat=train_feat,
            val_feat=val_feat,
            test_feat=test_feat,
            train_eval_feat=train_eval_feat,
            feature_columns=["f0"],
            all_items=all_items,
        )

    first_result = _run()
    assert _FitCountingModel.n_fit_calls == 3
    assert len(first_result) == 3

    # Simula uma interrupção: chamar de novo NÃO deve retreinar nenhum trial já logado.
    second_result = _run()
    assert (
        _FitCountingModel.n_fit_calls == 3
    ), "trials já logados não devem ser retreinados"
    assert {r[0] for r in second_result} == {r[0] for r in first_result}


def test_register_best_trial_picks_highest_test_ndcg_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_log_model_artifact(monkeypatch)

    with mlflow.start_run(run_name="fake-baseline"):
        mlflow.set_tags({"model_family": "fake", "feature_set": "fe_v4"})
        mlflow.log_metric("test_ndcg", 0.5)

    with mlflow.start_run(run_name="fake-tuned") as best_run:
        mlflow.set_tags({"model_family": "fake", "feature_set": "fe_v4"})
        mlflow.log_metric("test_ndcg", 0.8)
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression().fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
        mlflow.sklearn.log_model(model, name="model")

    version = register_best_trial(EXPERIMENT_NAME, "fake", "fake-registered-model")
    assert version is not None

    client = mlflow.MlflowClient()
    registered = client.get_model_version("fake-registered-model", version)
    assert registered.run_id == best_run.info.run_id
    assert registered.tags["model_family"] == "fake"

    # Idempotente: chamar de novo não deve criar uma segunda versão.
    version_again = register_best_trial(
        EXPERIMENT_NAME, "fake", "fake-registered-model"
    )
    assert version_again == version

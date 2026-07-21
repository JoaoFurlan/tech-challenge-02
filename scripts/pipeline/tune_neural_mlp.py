"""Tunagem do MLP: script standalone e resumível, fora do `dvc.yaml`.

Roda por último, de propósito (depois que o restante do pipeline já
reproduziu os números do sandbox). Duas buscas, mesmo espaço de busca e
mesma seed, comparando o efeito de duas melhorias sobre o baseline
default:

- `mlp-fe_v4-baseline`: configuração original (BCE sem peso de classe,
  `patience=5`) -- tag `trial_type=control`.
- `mlp-fe_v4-tuned`: `pos_weight` automático (compensa o desbalanceamento
  1:4 do negative sampling) + `patience=10` -- tag `trial_type=treatment`.

`control`/`treatment` (não `baseline`/`tuned`, como nos modelos tabulares)
porque isto é uma comparação pareada de verdade: os MESMOS 12 hiperparâmetros
amostrados (mesma seed) rodam nas duas buscas, variando só `pos_weight`/
`patience` -- isola o efeito dessa mudança especificamente, em vez de só
procurar o melhor config (que é o que `trial_type=search`/`baseline`/`tuned`
significam para os modelos tabulares).

Runs flat (sem categorias/nesting) -- ao final, compara TODOS os trials
`model_family=mlp` já logados (incluindo o baseline de `train_neural_mlp.py`)
e registra o de melhor `test_ndcg` no Model Registry.

Resumível de verdade (`run_hyperparameter_search` já verifica trial a
trial): interrompa a qualquer momento (`Ctrl+C`, kill, queda de energia) e
rode de novo -- os trials já logados são pulados, o script continua
exatamente de onde parou.

Uso:
    uv run python scripts/pipeline/tune_neural_mlp.py
"""

import sys
from typing import Any

import mlflow
from _common import EXPERIMENT_NAME, load_winning_feature_set

from recsys_ecommerce.config import load_training_config, settings
from recsys_ecommerce.features.pipeline import FeaturedTables
from recsys_ecommerce.models.neural_mlp import NeuralMLPModel
from recsys_ecommerce.tracking.mlflow_organization import (
    register_best_trial,
    run_hyperparameter_search,
)

if (reconfigure := getattr(sys.stdout, "reconfigure", None)) is not None:
    reconfigure(encoding="utf-8")

N_TRIALS = 12

# Mesmo espaço de busca nas duas buscas -- isola o efeito de pos_weight/patience
# do resto (arquitetura, regularização, otimização), já que o ParameterSampler
# com a mesma seed sorteia exatamente as mesmas combinações nas duas.
MLP_SEARCH_SPACE: dict[str, list[Any]] = {
    "hidden_dims": [(32,), (64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
    "dropout": [0.0, 0.1, 0.2, 0.3, 0.4],
    "lr": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
    "batch_size": [128, 256, 512, 1024],
}


def main() -> None:
    """Roda as duas buscas (control vs treatment) e registra o melhor trial do `mlp`."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    feature_set = load_winning_feature_set()
    tables = FeaturedTables.load(settings.data_dir, feature_set)

    run_hyperparameter_search(
        search_name="mlp-fe_v4-baseline",
        trial_type="control",
        model_class=NeuralMLPModel,
        search_space=MLP_SEARCH_SPACE,
        n_trials=N_TRIALS,
        model_family="mlp",
        feature_set=feature_set,
        experiment_name=EXPERIMENT_NAME,
        train_feat=tables.train,
        val_feat=tables.val,
        test_feat=tables.test,
        train_eval_feat=tables.train_eval,
        feature_columns=tables.feature_columns,
        all_items=tables.all_items,
        fixed_params={
            "max_epochs": 100,
            "patience": 5,
            "weighted_loss": False,
            "seed": 42,
        },
        seed=42,
    )

    run_hyperparameter_search(
        search_name="mlp-fe_v4-tuned",
        trial_type="treatment",
        model_class=NeuralMLPModel,
        search_space=MLP_SEARCH_SPACE,
        n_trials=N_TRIALS,
        model_family="mlp",
        feature_set=feature_set,
        experiment_name=EXPERIMENT_NAME,
        train_feat=tables.train,
        val_feat=tables.val,
        test_feat=tables.test,
        train_eval_feat=tables.train_eval,
        feature_columns=tables.feature_columns,
        all_items=tables.all_items,
        fixed_params={
            "max_epochs": 100,
            "patience": 10,
            "weighted_loss": True,
            "seed": 42,
        },
        seed=42,
    )

    cfg = load_training_config()
    register_best_trial(EXPERIMENT_NAME, "mlp", cfg.registered_model_name)


if __name__ == "__main__":
    main()

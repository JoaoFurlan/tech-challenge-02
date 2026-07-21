"""Configurações do projeto, carregadas de variáveis de ambiente ou `.env`."""

from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações de infraestrutura, carregadas de variáveis de ambiente ou `.env`.

    Centraliza valores que precisam ser consistentes entre os diferentes
    estágios do pipeline (pré-processamento, treino, tracking), evitando que
    fiquem espalhados e fixados diretamente no código (`hardcoded`).

    Attributes:
        random_seed: Semente global usada para reprodutibilidade (splits,
            inicialização de modelos, amostragem em baselines).
        data_dir: Diretório onde os dados brutos/processados ficam
            armazenados.
        models_dir: Diretório onde artefatos de modelo treinados são salvos
            localmente.
        mlflow_tracking_uri: URI do servidor de tracking do MLflow.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    random_seed: int = 42
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    mlflow_tracking_uri: str = "http://localhost:5000"


settings = Settings()


class TrainingConfig(BaseModel):
    """Hiperparâmetros de treino/avaliação, vencedores das buscas reais de tunagem.

    Diferente de `Settings` (configuração de infraestrutura, vinda de
    variáveis de ambiente), estes valores descrevem o experimento em si —
    lidos de um arquivo YAML versionado (`configs/model.yaml`), já que mudam
    por execução/experimento em vez de por ambiente. Os defaults abaixo são
    os hiperparâmetros vencedores das buscas aleatórias reais
    (`scripts/pipeline/tune_tabular_models.py`/`tune_neural_mlp.py`), não
    valores arbitrários -- ver `docs/model-card.md` para os números completos.

    Attributes:
        min_interactions: Limite mínimo de interações (usuário e item) para
            o filtro de cold-start no pré-processamento.
        negative_sampling_ratio: Número de exemplos negativos amostrados por
            exemplo positivo no split de treino.
        eval_negative_samples: Número de negativos amostrados por usuário na
            avaliação leave-one-out (convenção do paper NCF).
        top_k: Tamanho do ranking usado nas métricas (NDCG@k, Hit Rate@k,
            MRR@k, Coverage@k).
        registered_model_name: Nome do modelo no MLflow Model Registry.
        winning_feature_set: Nome do conjunto de features vencedor, decidido
            manualmente após revisar `scripts/experiments/run_fe_comparison.py`
            (ex.: `"fe_v4"`) -- ver `scripts/pipeline/promote_feature_set.py`
            para trocar. Usado por `feature_eng.py` (o que persistir) e por
            `model-selection` (`_common.load_winning_feature_set`, o que carregar).
        logreg_c: Inverso da força de regularização da regressão logística
            (`test_ndcg@10 = 0.6479`).
        logreg_solver: Solver da regressão logística.
        logreg_max_iter: Máximo de iterações do solver.
        decision_tree_max_depth: Profundidade máxima da árvore de decisão.
        decision_tree_min_samples_split: Mínimo de amostras para dividir um nó.
        decision_tree_min_samples_leaf: Mínimo de amostras por folha.
        xgboost_max_depth: Profundidade máxima das árvores do XGBoost.
        xgboost_learning_rate: Taxa de aprendizado do boosting.
        xgboost_n_estimators: Número de árvores (rounds de boosting).
        xgboost_subsample: Fração de amostras usada por árvore.
        xgboost_colsample_bytree: Fração de colunas usada por árvore.
        xgboost_reg_alpha: Regularização L1.
        xgboost_reg_lambda: Regularização L2.
        lightgbm_num_leaves: Número máximo de folhas por árvore.
        lightgbm_max_depth: Profundidade máxima das árvores.
        lightgbm_learning_rate: Taxa de aprendizado do boosting.
        lightgbm_n_estimators: Número de árvores (rounds de boosting).
        lightgbm_subsample: Fração de amostras usada por árvore.
        lightgbm_colsample_bytree: Fração de colunas usada por árvore.
        lightgbm_reg_alpha: Regularização L1.
        lightgbm_reg_lambda: Regularização L2.
        mlp_hidden_dims: Tamanhos das camadas ocultas do MLP.
        mlp_dropout: Taxa de dropout entre as camadas ocultas.
        mlp_lr: Taxa de aprendizado do otimizador Adam.
        mlp_weight_decay: Regularização L2 do otimizador Adam.
        mlp_batch_size: Tamanho do batch de treino.
        mlp_max_epochs: Número máximo de épocas de treino.
        mlp_patience: Épocas sem melhora na validação interna toleradas antes
            do early stopping.
        mlp_weighted_loss: Se `True`, pondera a BCE pelo desbalanceamento de
            classes do negative sampling (`pos_weight = n_neg / n_pos`).
        mlp_eval_every_n_epochs: Intervalo (em épocas) para reavaliar as
            métricas de ranking completas durante o treino do MLP, além da
            perda por época — usado para os gráficos de linha no MLflow.
    """

    min_interactions: int = 3
    negative_sampling_ratio: int = 4
    eval_negative_samples: int = 99
    top_k: int = 10
    registered_model_name: str = "recsys_ecommerce"
    winning_feature_set: str = "fe_v4"

    logreg_c: float = 1000.0
    logreg_solver: str = "lbfgs"
    logreg_max_iter: int = 1000

    decision_tree_max_depth: int = 5
    decision_tree_min_samples_split: int = 5
    decision_tree_min_samples_leaf: int = 5

    xgboost_max_depth: int = 4
    xgboost_learning_rate: float = 0.01
    xgboost_n_estimators: int = 50
    xgboost_subsample: float = 0.8
    xgboost_colsample_bytree: float = 0.6
    xgboost_reg_alpha: float = 0.0
    xgboost_reg_lambda: float = 0.1

    lightgbm_num_leaves: int = 15
    lightgbm_max_depth: int = 8
    lightgbm_learning_rate: float = 0.02
    lightgbm_n_estimators: int = 50
    lightgbm_subsample: float = 1.0
    lightgbm_colsample_bytree: float = 0.8
    lightgbm_reg_alpha: float = 0.01
    lightgbm_reg_lambda: float = 0.0

    mlp_hidden_dims: list[int] = [64, 32]
    mlp_dropout: float = 0.2
    mlp_lr: float = 0.001
    mlp_weight_decay: float = 0.0
    mlp_batch_size: int = 512
    mlp_max_epochs: int = 30
    mlp_patience: int = 5
    mlp_weighted_loss: bool = True
    mlp_eval_every_n_epochs: int = 5


def load_training_config(path: Path = Path("configs/model.yaml")) -> TrainingConfig:
    """Carrega os hiperparâmetros de treino/avaliação a partir de um YAML.

    Args:
        path: Caminho do arquivo YAML de configuração.

    Returns:
        Hiperparâmetros validados, com os defaults de `TrainingConfig` para
        qualquer chave ausente no arquivo.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return TrainingConfig(**raw)

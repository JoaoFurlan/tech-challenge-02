"""MLP em PyTorch sobre as features tabulares (`fe_v4`) — arquitetura decidida no sandbox.

Ao contrário do `neural_cf` de embeddings puros do projeto raiz (arquitetura
superada), este MLP consome as MESMAS features tabulares dos modelos de
árvore — o que permite reaproveitar `TabularClassifierModel` diretamente
(`NeuralMLPClassifier` já é sklearn-like), sem precisar de uma subclasse
própria de `RecommenderModel`.
"""

import random
from typing import Self

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from recsys_ecommerce.evaluation.metrics import evaluate_model
from recsys_ecommerce.models.tabular_classifier import TabularClassifierModel

SEED = 42
# PyTorch paraleliza operações de CPU (multiplicação de matrizes, redução de
# gradientes) entre threads por padrão, usando o número de cores que detecta
# na máquina -- em máquinas diferentes (ou containers com CPUs diferentes
# visíveis), isso muda a ordem de agregação ponto-flutuante das mesmas
# operações, produzindo resultados de treino ligeiramente diferentes mesmo
# com a mesma seed e o mesmo código (mesmo problema do XGBoost/LightGBM, ver
# xgboost_model.py). Fixar numa CONSTANTE (não `os.cpu_count()`, que é
# justamente o valor que difere entre máquinas) remove essa fonte de
# não-determinismo entre ambientes.
N_THREADS = 4


def _set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(N_THREADS)


class MLP(nn.Module):
    """Rede feed-forward simples: camadas ocultas configuráveis + saída sigmoid (via logits)."""

    def __init__(
        self,
        n_features: int,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
    ) -> None:
        """Monta as camadas: `Linear -> ReLU -> Dropout` por camada oculta, saída `Linear(., 1)`."""
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = n_features
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna os logits (antes do sigmoid) para cada linha de `x`."""
        result: torch.Tensor = self.net(x).squeeze(-1)
        return result


class NeuralMLPClassifier:
    """Wrapper sklearn-like (`fit`/`predict_proba`) em torno do `MLP`, porta do sandbox.

    `pos_weight` compensa o desbalanceamento do negative sampling (1:4,
    ~20% positivos) automaticamente a cada `fit()`. Early stopping usa uma
    fatia interna de validação (10% do próprio `X` passado a `fit()`),
    separada de `val_feat`/`test_feat` — evita otimizar duas vezes no mesmo
    conjunto usado para escolher hiperparâmetros.

    Reavaliação periódica (opcional, via `set_periodic_eval`): a cada
    `eval_every_n_epochs` épocas, recalcula NDCG/hit_rate/MRR/coverage
    completos em val/test e loga como step-metric no MLflow (além da perda
    por época, sempre logada) — só acontece se houver uma run ativa no
    momento de `fit()`, e nunca em testes/uso isolado.
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 512,
        max_epochs: int = 100,
        patience: int = 10,
        weighted_loss: bool = True,
        eval_every_n_epochs: int = 5,
        seed: int = SEED,
    ) -> None:
        """Guarda os hiperparâmetros (o treino de verdade só acontece em `fit`)."""
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.weighted_loss = weighted_loss
        self.eval_every_n_epochs = eval_every_n_epochs
        self.seed = seed
        self._periodic_eval: (
            tuple[pd.DataFrame, pd.DataFrame, list[str], np.ndarray] | None
        ) = None

    def get_params(self) -> dict[str, object]:
        """Hiperparâmetros do construtor, para logging no MLflow."""
        return {
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "weighted_loss": self.weighted_loss,
        }

    def set_periodic_eval(
        self,
        val_feat: pd.DataFrame,
        test_feat: pd.DataFrame,
        feature_columns: list[str],
        all_items: np.ndarray,
    ) -> Self:
        """Liga a reavaliação periódica de ranking durante o treino (opcional)."""
        self._periodic_eval = (val_feat, test_feat, feature_columns, all_items)
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """Treina o MLP. Ver `models.base.RecommenderModel.fit`."""
        _set_seed(self.seed)
        X_arr = np.asarray(X, dtype="float32")
        y_arr = np.asarray(y, dtype="float32")

        self.scaler_ = StandardScaler()
        self.scaler_.fit(X_arr)

        n = len(X_arr)
        rng = np.random.RandomState(self.seed)
        idx = rng.permutation(n)
        n_stop = max(1, int(n * 0.1))
        stop_idx, fit_idx = idx[:n_stop], idx[n_stop:]

        X_fit = self.scaler_.transform(X_arr[fit_idx])
        X_stop = self.scaler_.transform(X_arr[stop_idx])
        y_fit, y_stop = y_arr[fit_idx], y_arr[stop_idx]

        loss_fn = self._build_loss_fn(y_fit)

        n_features = X_fit.shape[1]
        self.model_ = MLP(n_features, self.hidden_dims, self.dropout)
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_fit), torch.from_numpy(y_fit)
        )
        generator = torch.Generator().manual_seed(self.seed)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, generator=generator
        )
        X_stop_t, y_stop_t = torch.from_numpy(X_stop), torch.from_numpy(y_stop)

        self._train_loop(loader, len(dataset), X_stop_t, y_stop_t, loss_fn, optimizer)
        return self

    def _build_loss_fn(self, y_fit: np.ndarray) -> nn.Module:
        if not self.weighted_loss:
            self.pos_weight_ = 1.0
            return nn.BCEWithLogitsLoss()
        n_pos = y_fit.sum()
        n_neg = len(y_fit) - n_pos
        self.pos_weight_ = float(n_neg / max(n_pos, 1.0))
        return nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(self.pos_weight_, dtype=torch.float32)
        )

    def _train_loop(
        self,
        loader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
        n_samples: int,
        X_stop_t: torch.Tensor,
        y_stop_t: torch.Tensor,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        active_run = mlflow.active_run() is not None
        self.loss_history_: dict[str, list[float]] = {"train": [], "early_stop": []}
        best_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(self.max_epochs):
            train_loss = self._train_one_epoch(loader, n_samples, loss_fn, optimizer)
            stop_loss = self._eval_loss(X_stop_t, y_stop_t, loss_fn)
            self.loss_history_["train"].append(train_loss)
            self.loss_history_["early_stop"].append(stop_loss)

            if active_run:
                self._log_epoch_metrics(epoch, train_loss, stop_loss)

            if stop_loss < best_loss - 1e-5:
                best_loss = stop_loss
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.n_epochs_trained_ = len(self.loss_history_["train"])

    def _train_one_epoch(
        self,
        loader: torch.utils.data.DataLoader[tuple[torch.Tensor, ...]],
        n_samples: int,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        self.model_.train()
        running_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(self.model_(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(xb)
        return running_loss / n_samples

    def _eval_loss(
        self, X_t: torch.Tensor, y_t: torch.Tensor, loss_fn: nn.Module
    ) -> float:
        self.model_.eval()
        with torch.no_grad():
            loss_value: float = loss_fn(self.model_(X_t), y_t).item()
            return loss_value

    def _log_epoch_metrics(
        self, epoch: int, train_loss: float, stop_loss: float
    ) -> None:
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("early_stop_loss", stop_loss, step=epoch)
        if self._periodic_eval and epoch % self.eval_every_n_epochs == 0:
            val_feat, _test_feat, feature_columns, all_items = self._periodic_eval
            val_metrics = evaluate_model(self, val_feat, feature_columns, all_items)
            for name, value in val_metrics.items():
                mlflow.log_metric(f"val_{name}", value, step=epoch)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Retorna P(interação) por linha. Ver `models.base.RecommenderModel.predict_proba`."""
        X_arr = np.asarray(X, dtype="float32")
        X_scaled = self.scaler_.transform(X_arr)
        self.model_.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self.model_(torch.from_numpy(X_scaled))).numpy()
        return np.column_stack([1 - probs, probs])


class NeuralMLPModel(TabularClassifierModel):
    """MLP sobre `fe_v4`, encaixado em `TabularClassifierModel` via `NeuralMLPClassifier`."""

    def __init__(self, **kwargs: object) -> None:
        """Inicializa o modelo.

        Args:
            **kwargs: Repassados a `NeuralMLPClassifier` (ex.: `hidden_dims`,
                `dropout`, `lr`, `patience`, `weighted_loss`).
        """
        super().__init__(NeuralMLPClassifier(**kwargs))  # type: ignore[arg-type]

    def set_periodic_eval(
        self,
        val_feat: pd.DataFrame,
        test_feat: pd.DataFrame,
        feature_columns: list[str],
        all_items: np.ndarray,
    ) -> Self:
        """Liga a reavaliação periódica de ranking durante o treino (ver `NeuralMLPClassifier`)."""
        self._clf.set_periodic_eval(val_feat, test_feat, feature_columns, all_items)
        return self

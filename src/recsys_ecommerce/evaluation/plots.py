"""Gráficos de avaliação, logados como artefatos no MLflow (porta direta do sandbox)."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_metrics_across_splits(
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    title: str,
) -> Figure:
    """Compara as métricas de um modelo entre os 3 splits, uma barra por métrica.

    Args:
        train_metrics: Métricas no split de treino (diagnóstico).
        val_metrics: Métricas no split de validação.
        test_metrics: Métricas no split de teste.
        title: Título do gráfico.

    Returns:
        A figura montada (chame `plt.close(fig)` após logar).
    """
    metric_names = list(train_metrics.keys())
    fig, axes = plt.subplots(1, len(metric_names), figsize=(4 * len(metric_names), 3.5))
    for ax, name in zip(axes, metric_names, strict=True):
        values = [train_metrics[name], val_metrics[name], test_metrics[name]]
        ax.bar(
            ["train", "val", "test"], values, color=["#4C72B0", "#DD8452", "#55A868"]
        )
        ax.set_title(name)
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_feature_importance(
    model: Any, feature_columns: list[str], model_name: str
) -> Figure:  # noqa: ANN401
    """Importância de features do modelo (baseada em ganho/impureza, ou `|coeficiente|`).

    Args:
        model: Estimador sklearn-compatible já treinado (ex.:
            `TabularClassifierModel.underlying_estimator`).
        feature_columns: Nomes das colunas de features, na ordem usada no treino.
        model_name: Nome do modelo (para o título).

    Returns:
        A figura montada (chame `plt.close(fig)` após logar).
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        ylabel = "importância (baseada em impureza/ganho)"
    else:
        # Regressão logística: |coeficiente| não é diretamente comparável em escala
        # com importância de árvore (features aqui não são padronizadas) -- serve
        # só pra ver a direção/peso relativo DENTRO deste modelo.
        importances = np.abs(model.coef_[0])
        ylabel = "|coeficiente| (não comparável em escala com árvores)"

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(feature_columns, importances)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Importância de features -- {model_name}")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    return fig

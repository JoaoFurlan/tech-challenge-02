"""Engenharia de features tabulares, em camadas (`fe_v1` a `fe_v4`).

Cada versão adiciona colunas à anterior — a mesma composição em camadas
validada no sandbox (`new_repo/notebooks/01_eda.ipynb`). `fe_v4` é a versão
vencedora, usada pela etapa `feature_eng` do `dvc.yaml`; as versões
anteriores existem para o script de comparação
(`scripts/experiments/run_fe_comparison.py`) recriar a mesma superfície de
comparação que já existe no MLflow do sandbox.
"""

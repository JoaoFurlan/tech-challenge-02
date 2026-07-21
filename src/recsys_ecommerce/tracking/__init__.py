"""Organização de runs no MLflow: categorias, sub-grupos e leaderboard.

Porta para código testável a organização validada no sandbox
(`new_repo/notebooks/`): uma run de "categoria" por família de modelo
(`tree-baseline-models`, `neural-models`), sub-grupos aninhados
(`feature-engineering`, `hyperparameter-tuning`), buscas de hiperparâmetros
que logam cada trial por completo (sem uma run "campeão" separada), e um
mecanismo de leaderboard que loga o melhor resultado direto nas runs de
categoria/sub-grupo.
"""

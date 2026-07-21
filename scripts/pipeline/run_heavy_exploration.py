"""Roda toda a exploração "pesada" do projeto, de uma vez: FE completa + as 2 tunagens reais.

Contraparte do caminho rápido (`dvc repro`, só o necessário para reproduzir
o pipeline): isto é o que foi de fato usado para descobrir os números finais
(mostrado no vídeo STAR), não o que se espera que um avaliador rode. Cada
um dos 3 scripts já é resumível individualmente -- interromper este wrapper
e rodar de novo não perde nada, cada etapa pula o que já está feito.

Uso:
    uv run python scripts/pipeline/run_heavy_exploration.py
"""

import subprocess

HEAVY_SCRIPTS = [
    "scripts/experiments/run_fe_comparison.py",
    "scripts/pipeline/tune_tabular_models.py",
    "scripts/pipeline/tune_neural_mlp.py",
]


def main() -> None:
    """Roda os 3 scripts pesados em sequência, parando no primeiro que falhar."""
    for script in HEAVY_SCRIPTS:
        print(f"\n=== {script} ===")
        result = subprocess.run(["uv", "run", "python", script], check=False)
        if result.returncode != 0:
            raise SystemExit(
                f"{script} falhou (exit code {result.returncode}) -- interrompendo."
            )

    print(
        "\nExploração pesada completa. Rode `promote_best_model.py` para refletir os resultados."
    )


if __name__ == "__main__":
    main()

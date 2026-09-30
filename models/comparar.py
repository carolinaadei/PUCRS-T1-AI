"""
Item 5 do enunciado — comparação entre os algoritmos e escolha do melhor.

Executa os cinco algoritmos nas duas abordagens de pré-processamento e monta a
tabela e o gráfico comparativos a partir de `reports/metrics/resultados.csv`.

Execução:
    python models/comparar.py              # roda tudo e gera a comparação
    python models/comparar.py --somente-tabela   # só relê o CSV já existente
"""

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ttt import evaluation, preprocessing
from ttt.config import FIGURES_DIR, METRICS_DIR, garantir_diretorios
from ttt.experiment import executar

# Módulo de cada algoritmo, na ordem em que aparecem no relatório
ALGORITMOS = [
    "models.knn",
    "models.arvore_decisao",
    "models.mlp",
    "models.svm",
    "models.xgboost_clf",
]


def rodar_todos() -> None:
    """Executa cada algoritmo em cada abordagem, registrando os resultados."""
    for caminho in ALGORITMOS:
        try:
            modulo = importlib.import_module(caminho)
        except ImportError as erro:
            # XGBoost é dependência opcional; não deve derrubar a comparação
            print(f"[AVISO] {caminho} ignorado — dependência ausente ({erro.name}).")
            continue

        for abordagem in preprocessing.ABORDAGENS:
            executar(
                nome=modulo.NOME,
                modelo=modulo.criar_modelo(),
                param_grid=modulo.PARAM_GRID,
                abordagem=abordagem,
            )


def montar_tabela() -> None:
    """Imprime a tabela comparativa e salva o gráfico de barras."""
    df = evaluation.carregar_resultados()
    if df.empty:
        print("[AVISO] Nenhum resultado registrado. Rode os algoritmos primeiro.")
        return

    df = df.sort_values("acuracia_teste", ascending=False)

    print("\n" + "=" * 78)
    print("COMPARAÇÃO DOS ALGORITMOS — conjunto de teste".center(78))
    print("=" * 78)
    print(df[[
        "algoritmo", "abordagem", "acuracia_val", "acuracia_teste",
        "precision", "recall", "f1",
    ]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    melhor = df.iloc[0]
    print("\n" + "-" * 78)
    print(f"Melhor configuração: {melhor['algoritmo']} com abordagem '{melhor['abordagem']}'")
    print(f"  Acurácia no teste : {melhor['acuracia_teste']:.4f}")
    print(f"  F-measure         : {melhor['f1']:.4f}")
    print(f"  Hiperparâmetros   : {melhor['melhores_params']}")
    print("-" * 78)

    _comparar_abordagens(df)
    _salvar_grafico(df)

    destino = METRICS_DIR / "comparacao_final.csv"
    df.to_csv(destino, index=False)
    print(f"[CSV] {destino.relative_to(METRICS_DIR.parent.parent)}")


def _comparar_abordagens(df) -> None:
    """Responde à pergunta do item 3: qual abordagem é mais adequada."""
    if df["abordagem"].nunique() < 2:
        return

    medias = df.groupby("abordagem")["acuracia_teste"].mean().sort_values(ascending=False)
    print("\nAcurácia média por abordagem de pré-processamento:")
    for abordagem, media in medias.items():
        print(f"  {abordagem:<10} {media:.4f}  ({preprocessing.obter(abordagem).descricao})")
    print(f"  → Abordagem mais adequada no geral: '{medias.index[0]}'")


def _salvar_grafico(df) -> None:
    garantir_diretorios()

    abordagens = sorted(df["abordagem"].unique())
    algoritmos = list(dict.fromkeys(df["algoritmo"]))
    x = np.arange(len(algoritmos))
    largura = 0.8 / len(abordagens)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, abordagem in enumerate(abordagens):
        sub = df[df["abordagem"] == abordagem].set_index("algoritmo")
        # reindex mantém a ordem das barras alinhada mesmo se faltar um algoritmo
        valores = sub["acuracia_teste"].reindex(algoritmos)
        pos = x + (i - (len(abordagens) - 1) / 2) * largura
        barras = ax.bar(pos, valores.fillna(0), largura, label=abordagem)
        for barra, valor in zip(barras, valores):
            if not np.isnan(valor):
                ax.text(barra.get_x() + barra.get_width() / 2, valor + 0.015,
                        f"{valor:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(algoritmos, rotation=12, ha="right")
    ax.set_ylabel("Acurácia no conjunto de teste")
    ax.set_title("Comparação dos algoritmos por abordagem de pré-processamento")
    ax.set_ylim(0, 1.1)
    ax.legend(title="Abordagem")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    destino = FIGURES_DIR / "comparacao_algoritmos.png"
    fig.savefig(destino, dpi=140)
    plt.close(fig)
    print(f"[FIG] {destino.relative_to(FIGURES_DIR.parent.parent)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--somente-tabela",
        action="store_true",
        help="Não reexecuta os algoritmos; apenas relê os resultados já registrados",
    )
    args = parser.parse_args()

    if not args.somente_tabela:
        rodar_todos()
    montar_tabela()

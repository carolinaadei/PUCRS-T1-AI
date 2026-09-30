"""
Comparação entre os algoritmos e escolha do melhor (item 5 do enunciado).

Roda os cinco algoritmos nas duas abordagens e monta a tabela e o gráfico
comparativos a partir de `reports/metrics/resultados.csv`.

    python models/comparar.py                  # roda tudo e gera a comparação
    python models/comparar.py --somente-tabela # só relê o CSV já existente
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from models import arvore_decisao, knn, mlp, svm
from ttt import evaluation, preprocessing
from ttt.config import FIGURES_DIR, METRICS_DIR
from ttt.experiment import treinar

ALGORITMOS = [knn, arvore_decisao, mlp, svm]

# XGBoost é dependência opcional: sua ausência não derruba a comparação
try:
    from models import xgboost_clf
    ALGORITMOS.append(xgboost_clf)
except ImportError:
    print("[AVISO] xgboost não instalado — algoritmo ignorado na comparação.")


def rodar_todos():
    """Executa cada algoritmo em cada abordagem, registrando os resultados."""
    for algoritmo in ALGORITMOS:
        for abordagem in preprocessing.ABORDAGENS:
            treinar(algoritmo.NOME, algoritmo.MODELO, algoritmo.PARAM_GRID, abordagem)


def montar_tabela():
    """Imprime a tabela comparativa e salva o gráfico de barras."""
    df = evaluation.carregar_resultados()
    if df.empty:
        print("[AVISO] Nenhum resultado registrado. Rode os algoritmos primeiro.")
        return

    df = df.sort_values("acuracia_teste", ascending=False)

    print("\n" + "=" * 78)
    print("COMPARAÇÃO DOS ALGORITMOS — conjunto de teste".center(78))
    print("=" * 78)
    print(df.drop(columns="melhores_params").to_string(
        index=False, float_format=lambda v: f"{v:.4f}"
    ))

    melhor = df.iloc[0]
    print("\n" + "-" * 78)
    print(f"Melhor configuração: {melhor['algoritmo']} com abordagem '{melhor['abordagem']}'")
    print(f"  Acurácia no teste : {melhor['acuracia_teste']:.4f}")
    print(f"  F-measure         : {melhor['f1']:.4f}")
    print(f"  Hiperparâmetros   : {melhor['melhores_params']}")
    print("-" * 78)

    _comparar_abordagens(df)
    _salvar_grafico(df)

    df.to_csv(METRICS_DIR / "comparacao_final.csv", index=False)
    print("[CSV] reports/metrics/comparacao_final.csv")


def _comparar_abordagens(df):
    """Qual abordagem de pré-processamento foi mais adequada, na média (item 3)."""
    if df["abordagem"].nunique() < 2:
        return

    medias = df.groupby("abordagem")["acuracia_teste"].mean().sort_values(ascending=False)
    print("\nAcurácia média por abordagem de pré-processamento:")
    for abordagem, media in medias.items():
        print(f"  {abordagem:<10} {media:.4f}  ({preprocessing.ABORDAGENS[abordagem]})")
    print(f"  → Abordagem mais adequada no geral: '{medias.index[0]}'")


def _salvar_grafico(df):
    abordagens = sorted(df["abordagem"].unique())
    algoritmos = list(dict.fromkeys(df["algoritmo"]))
    x = np.arange(len(algoritmos))
    largura = 0.8 / len(abordagens)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, abordagem in enumerate(abordagens):
        # reindex alinha as barras mesmo quando falta um algoritmo
        valores = (df[df["abordagem"] == abordagem]
                   .set_index("algoritmo")["acuracia_teste"]
                   .reindex(algoritmos))
        posicoes = x + (i - (len(abordagens) - 1) / 2) * largura

        ax.bar(posicoes, valores.fillna(0), largura, label=abordagem)
        for pos, valor in zip(posicoes, valores):
            if not np.isnan(valor):
                ax.text(pos, valor + 0.015, f"{valor:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(algoritmos, rotation=12, ha="right")
    ax.set_ylabel("Acurácia no conjunto de teste")
    ax.set_title("Comparação dos algoritmos por abordagem de pré-processamento")
    ax.set_ylim(0, 1.1)
    ax.legend(title="Abordagem")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    fig.savefig(FIGURES_DIR / "comparacao_algoritmos.png", dpi=140)
    plt.close(fig)
    print("[FIG] reports/figures/comparacao_algoritmos.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compara os algoritmos do trabalho.")
    parser.add_argument(
        "--somente-tabela",
        action="store_true",
        help="Não reexecuta os algoritmos; apenas relê os resultados já registrados",
    )
    args = parser.parse_args()

    if not args.somente_tabela:
        rodar_todos()
    montar_tabela()

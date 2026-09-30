"""
Avaliação padronizada dos modelos.

O enunciado (item 5) pede acurácia, precision, recall e F-measure para todos os
algoritmos e, ao final, uma comparação entre eles. Para que essa comparação
signifique alguma coisa, todos são medidos da mesma forma, sobre o mesmo
conjunto de teste.

Cada execução registra uma linha em `reports/metrics/resultados.csv`, e
`models/comparar.py` transforma esse arquivo na tabela e no gráfico finais.
"""

import matplotlib
matplotlib.use("Agg")  # backend não-interativo: salva sem abrir janela

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from ttt.config import CLASS_NAMES, FIGURES_DIR, METRICS_DIR

RESULTADOS_CSV = METRICS_DIR / "resultados.csv"

TODAS_AS_CLASSES = list(range(len(CLASS_NAMES)))


def avaliar(nome, abordagem, y_teste, y_pred, acuracia_val, melhores_params):
    """
    Calcula as métricas do enunciado, imprime o relatório, salva a matriz de
    confusão e registra o resultado no CSV consolidado.

    As médias são `weighted` porque as classes não estão balanceadas: "Empate"
    tem bem menos amostras que as demais, e uma média macro daria a essa classe
    minoritária o mesmo peso das outras quatro.

    Devolve um dicionário com as métricas.
    """
    resultado = {
        "algoritmo": nome,
        "abordagem": abordagem,
        "acuracia_val": float(acuracia_val),
        "acuracia_teste": accuracy_score(y_teste, y_pred),
        "precision": precision_score(y_teste, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_teste, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_teste, y_pred, average="weighted", zero_division=0),
        "melhores_params": str(melhores_params),
    }

    print("\n" + "=" * 70)
    print(f" {nome} — abordagem '{abordagem}' ".center(70, "="))
    print("=" * 70)
    print(f"Melhores parâmetros : {melhores_params}")
    print(f"Acurácia (validação): {resultado['acuracia_val']:.4f}")
    print(f"Acurácia (teste)    : {resultado['acuracia_teste']:.4f}")
    print(f"Precision (weighted): {resultado['precision']:.4f}")
    print(f"Recall    (weighted): {resultado['recall']:.4f}")
    print(f"F-measure (weighted): {resultado['f1']:.4f}\n")

    # `labels` explícito evita quebra quando alguma classe não é predita
    print(classification_report(
        y_teste, y_pred,
        labels=TODAS_AS_CLASSES,
        target_names=CLASS_NAMES,
        zero_division=0,
    ))

    _salvar_matriz_confusao(nome, abordagem, y_teste, y_pred)
    _registrar(resultado)
    return resultado


def carregar_resultados():
    """Lê o CSV consolidado; devolve um DataFrame vazio se nada foi executado."""
    if not RESULTADOS_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(RESULTADOS_CSV)


def _salvar_matriz_confusao(nome, abordagem, y_teste, y_pred):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_teste, y_pred,
        labels=TODAS_AS_CLASSES,
        display_labels=CLASS_NAMES,
        xticks_rotation=45,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"Matriz de Confusão — {nome} ({abordagem})")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    fig.tight_layout()

    destino = FIGURES_DIR / f"{_slug(nome)}_{abordagem}_confusao.png"
    fig.savefig(destino, dpi=140)
    plt.close(fig)
    print(f"[FIG] reports/figures/{destino.name}")


def _registrar(resultado):
    """
    Acrescenta o resultado ao CSV, substituindo a linha anterior do mesmo par
    (algoritmo, abordagem) para que reexecuções não acumulem lixo.
    """
    df = carregar_resultados()
    if not df.empty:
        df = df[~(
            (df["algoritmo"] == resultado["algoritmo"])
            & (df["abordagem"] == resultado["abordagem"])
        )]

    df = pd.concat([df, pd.DataFrame([resultado])], ignore_index=True)
    df.sort_values(["algoritmo", "abordagem"]).to_csv(RESULTADOS_CSV, index=False)
    print("[CSV] reports/metrics/resultados.csv")


def _slug(texto):
    return texto.lower().replace(" ", "_").replace("-", "_")

"""
Métricas do item 5: acurácia, precision, recall e F-measure.

Cada execução registra uma linha em `reports/metrics/resultados.csv`, que
`models/comparar.py` transforma na tabela e no gráfico finais.
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


def avaliar(nome, abordagem, y_teste, y_pred, acuracia_val, melhores_params,
            tempo_treino, tempo_predicao):
    """
    Mede o modelo no conjunto de teste, imprime o relatório, salva a matriz de
    confusão e registra o resultado no CSV. Devolve as métricas em um dicionário.

    As médias são `weighted` porque as classes estão desbalanceadas: "Empate"
    tem bem menos amostras que as demais, e uma média macro daria a ela o mesmo
    peso das outras quatro.

    Os tempos respondem ao "menos custosa" do item 3: `tempo_treino` é o ajuste
    do modelo escolhido e `tempo_predicao` cobre o conjunto de teste inteiro.
    """
    resultado = {
        "algoritmo": nome,
        "abordagem": abordagem,
        "acuracia_val": float(acuracia_val),
        "acuracia_teste": accuracy_score(y_teste, y_pred),
        "precision": precision_score(y_teste, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_teste, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_teste, y_pred, average="weighted", zero_division=0),
        "tempo_treino_s": round(tempo_treino, 4),
        "tempo_predicao_ms": round(tempo_predicao * 1000, 2),
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
    print(f"F-measure (weighted): {resultado['f1']:.4f}")
    print(f"Custo               : treino {resultado['tempo_treino_s']:.4f}s | "
          f"predição {resultado['tempo_predicao_ms']:.2f}ms\n")

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
    """Lê o CSV consolidado, ou um DataFrame vazio se nada foi executado ainda."""
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
    """Grava no CSV, substituindo a linha anterior do mesmo algoritmo e abordagem."""
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

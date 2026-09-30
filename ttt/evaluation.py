"""
Avaliação padronizada dos modelos.

O enunciado (item 5) pede acurácia, precision, recall e F-measure para todos os
algoritmos e, ao final, uma comparação entre eles. Para que essa comparação
signifique alguma coisa, todos precisam ser medidos da mesma forma, sobre o
mesmo conjunto de teste.

Cada execução de um algoritmo registra uma linha em
`reports/metrics/resultados.csv`, e `models/comparar.py` transforma esse
arquivo na tabela e no gráfico comparativos.
"""

from dataclasses import asdict, dataclass

import matplotlib
matplotlib.use("Agg")  # backend não-interativo: salva sem abrir janela

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from ttt.config import CLASS_NAMES, FIGURES_DIR, METRICS_DIR, garantir_diretorios

RESULTADOS_CSV = METRICS_DIR / "resultados.csv"

# Colunas do registro consolidado, na ordem em que aparecem no CSV
_COLUNAS = [
    "algoritmo", "abordagem", "acuracia_val", "acuracia_teste",
    "precision", "recall", "f1", "melhores_params",
]


@dataclass
class Resultado:
    """Métricas de um algoritmo sob uma abordagem de pré-processamento."""

    algoritmo: str
    abordagem: str
    acuracia_val: float
    acuracia_teste: float
    precision: float
    recall: float
    f1: float
    melhores_params: str

    def linha_resumo(self) -> str:
        return (
            f"{self.algoritmo} [{self.abordagem}] — "
            f"val: {self.acuracia_val:.4f} | teste: {self.acuracia_teste:.4f} | "
            f"F1: {self.f1:.4f}"
        )


def calcular(
    algoritmo: str,
    abordagem: str,
    y_teste: np.ndarray,
    y_pred: np.ndarray,
    acuracia_val: float,
    melhores_params: dict | str = "",
) -> Resultado:
    """
    Calcula as quatro métricas do enunciado sobre o conjunto de teste.

    As médias são `weighted` porque as classes não estão perfeitamente
    balanceadas — "Empate" tem bem menos amostras que as demais, e uma média
    macro daria a essa classe minoritária o mesmo peso das outras quatro.
    """
    return Resultado(
        algoritmo=algoritmo,
        abordagem=abordagem,
        acuracia_val=float(acuracia_val),
        acuracia_teste=float(accuracy_score(y_teste, y_pred)),
        precision=float(precision_score(y_teste, y_pred, average="weighted", zero_division=0)),
        recall=float(recall_score(y_teste, y_pred, average="weighted", zero_division=0)),
        f1=float(f1_score(y_teste, y_pred, average="weighted", zero_division=0)),
        melhores_params=str(melhores_params),
    )


def imprimir_relatorio(resultado: Resultado, y_teste: np.ndarray, y_pred: np.ndarray) -> None:
    """Imprime no console o relatório detalhado por classe."""
    titulo = f" {resultado.algoritmo} — abordagem '{resultado.abordagem}' "
    print("\n" + "=" * 70)
    print(titulo.center(70, "="))
    print("=" * 70)
    print(f"Melhores parâmetros : {resultado.melhores_params}")
    print(f"Acurácia (validação): {resultado.acuracia_val:.4f}")
    print(f"Acurácia (teste)    : {resultado.acuracia_teste:.4f}")
    print(f"Precision (weighted): {resultado.precision:.4f}")
    print(f"Recall    (weighted): {resultado.recall:.4f}")
    print(f"F-measure (weighted): {resultado.f1:.4f}\n")

    # `labels` explícito evita quebra quando alguma classe não é predita
    print(classification_report(
        y_teste, y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0,
    ))


def salvar_matriz_confusao(resultado: Resultado, y_teste: np.ndarray, y_pred: np.ndarray) -> None:
    """Salva a matriz de confusão em `reports/figures/<algoritmo>_<abordagem>.png`."""
    garantir_diretorios()

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_teste, y_pred,
        labels=list(range(len(CLASS_NAMES))),
        display_labels=CLASS_NAMES,
        xticks_rotation=45,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"Matriz de Confusão — {resultado.algoritmo} ({resultado.abordagem})")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    fig.tight_layout()

    destino = FIGURES_DIR / f"{_slug(resultado.algoritmo)}_{resultado.abordagem}_confusao.png"
    fig.savefig(destino, dpi=140)
    plt.close(fig)
    print(f"[FIG] {destino.relative_to(FIGURES_DIR.parent.parent)}")


def registrar(resultado: Resultado) -> None:
    """
    Acrescenta o resultado ao CSV consolidado, substituindo a linha anterior do
    mesmo par (algoritmo, abordagem) para que reexecuções não acumulem lixo.
    """
    garantir_diretorios()

    novo = pd.DataFrame([asdict(resultado)])[_COLUNAS]
    if RESULTADOS_CSV.exists():
        antigo = pd.read_csv(RESULTADOS_CSV)
        antigo = antigo[~(
            (antigo["algoritmo"] == resultado.algoritmo)
            & (antigo["abordagem"] == resultado.abordagem)
        )]
        novo = pd.concat([antigo, novo], ignore_index=True)

    novo.sort_values(["algoritmo", "abordagem"]).to_csv(RESULTADOS_CSV, index=False)
    print(f"[CSV] {RESULTADOS_CSV.relative_to(METRICS_DIR.parent.parent)}")


def carregar_resultados() -> pd.DataFrame:
    """Lê o CSV consolidado; devolve um DataFrame vazio se nada foi executado."""
    if not RESULTADOS_CSV.exists():
        return pd.DataFrame(columns=_COLUNAS)
    return pd.read_csv(RESULTADOS_CSV)


def finalizar(resultado: Resultado, y_teste: np.ndarray, y_pred: np.ndarray) -> Resultado:
    """Atalho para o encerramento padrão de todo script de algoritmo."""
    imprimir_relatorio(resultado, y_teste, y_pred)
    salvar_matriz_confusao(resultado, y_teste, y_pred)
    registrar(resultado)
    return resultado


def _slug(texto: str) -> str:
    return texto.lower().replace(" ", "_").replace("-", "_")

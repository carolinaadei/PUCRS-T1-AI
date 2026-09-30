"""
Carga dos conjuntos de treino, validação e teste.

Os três arquivos são a divisão física exigida pelo enunciado (item 4) e são
gerados por `notebooks/01_construcao_dataset.ipynb`. Todos os algoritmos devem
carregar os dados por aqui, para que os experimentos sejam comparáveis entre si.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ttt.config import (
    FEATURE_COLS,
    TARGET_COL,
    TESTE_CSV,
    TREINO_CSV,
    VALIDACAO_CSV,
)


@dataclass
class Split:
    """Um conjunto de dados: features (DataFrame) e alvo (array de inteiros)."""

    X: pd.DataFrame
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class Dados:
    """Os três conjuntos da divisão física."""

    treino: Split
    validacao: Split
    teste: Split

    def resumo(self) -> str:
        return (
            f"Treino: {len(self.treino)} | "
            f"Validação: {len(self.validacao)} | "
            f"Teste: {len(self.teste)}"
        )

    def treino_mais_validacao(self) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Concatena treino + validação e devolve também o vetor `test_fold` do
        `PredefinedSplit`: -1 marca as linhas de treino (nunca usadas para
        pontuar) e 0 marca as de validação.

        É assim que o GridSearchCV usa o conjunto de validação físico como
        único fold, em vez de reparticionar o treino aleatoriamente.
        """
        X = pd.concat([self.treino.X, self.validacao.X], ignore_index=True)
        y = np.concatenate([self.treino.y, self.validacao.y])
        test_fold = np.concatenate([
            np.full(len(self.treino), -1, dtype=int),
            np.zeros(len(self.validacao), dtype=int),
        ])
        return X, y, test_fold


def _ler(caminho) -> Split:
    df = pd.read_csv(caminho)

    faltando = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if faltando:
        raise ValueError(f"{caminho.name}: colunas ausentes {faltando}")

    return Split(X=df[FEATURE_COLS].copy(), y=df[TARGET_COL].to_numpy())


def carregar() -> Dados:
    """Lê os três CSVs processados a partir da raiz do repositório."""
    return Dados(
        treino=_ler(TREINO_CSV),
        validacao=_ler(VALIDACAO_CSV),
        teste=_ler(TESTE_CSV),
    )

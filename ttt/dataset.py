"""
Carga da divisão treino / validação / teste (item 4 do enunciado).

Os três arquivos são gerados por `notebooks/01_construcao_dataset.ipynb`.
"""

from collections import namedtuple

import numpy as np
import pandas as pd

from ttt.config import FEATURE_COLS, TARGET_COL, TESTE_CSV, TREINO_CSV, VALIDACAO_CSV

Dados = namedtuple("Dados", "X_treino y_treino X_val y_val X_teste y_teste")


def carregar():
    """Lê os três CSVs e separa as features do alvo."""

    def ler(caminho):
        df = pd.read_csv(caminho)
        return df[FEATURE_COLS], df[TARGET_COL].to_numpy()

    X_treino, y_treino = ler(TREINO_CSV)
    X_val, y_val = ler(VALIDACAO_CSV)
    X_teste, y_teste = ler(TESTE_CSV)

    return Dados(X_treino, y_treino, X_val, y_val, X_teste, y_teste)


def juntar_treino_validacao(dados):
    """
    Junta treino e validação para o GridSearchCV, junto do `test_fold` do
    `PredefinedSplit`: -1 nas linhas de treino, 0 nas de validação.

    Assim a busca pontua no conjunto de validação físico, em vez de
    reparticionar o treino aleatoriamente.
    """
    X = pd.concat([dados.X_treino, dados.X_val], ignore_index=True)
    y = np.concatenate([dados.y_treino, dados.y_val])
    test_fold = np.concatenate([
        np.full(len(dados.y_treino), -1),
        np.zeros(len(dados.y_val), dtype=int),
    ])
    return X, y, test_fold

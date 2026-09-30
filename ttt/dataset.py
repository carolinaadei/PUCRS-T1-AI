"""
Carga dos conjuntos de treino, validação e teste.

Os três arquivos são a divisão física exigida pelo enunciado (item 4) e são
gerados por `notebooks/01_construcao_dataset.ipynb`. Todos os algoritmos
carregam os dados por aqui, para que os experimentos sejam comparáveis.
"""

from collections import namedtuple

import numpy as np
import pandas as pd

from ttt.config import FEATURE_COLS, TARGET_COL, TESTE_CSV, TREINO_CSV, VALIDACAO_CSV

Dados = namedtuple("Dados", "X_treino y_treino X_val y_val X_teste y_teste")


def carregar():
    """Lê os três CSVs processados e separa features do alvo."""

    def ler(caminho):
        df = pd.read_csv(caminho)
        return df[FEATURE_COLS], df[TARGET_COL].to_numpy()

    X_treino, y_treino = ler(TREINO_CSV)
    X_val, y_val = ler(VALIDACAO_CSV)
    X_teste, y_teste = ler(TESTE_CSV)

    return Dados(X_treino, y_treino, X_val, y_val, X_teste, y_teste)


def juntar_treino_validacao(dados):
    """
    Concatena treino + validação e monta o vetor `test_fold` do
    `PredefinedSplit`: -1 marca as linhas de treino (nunca usadas para pontuar)
    e 0 marca as de validação.

    É assim que o GridSearchCV usa o conjunto de validação físico como único
    fold, em vez de reparticionar o treino aleatoriamente.
    """
    X = pd.concat([dados.X_treino, dados.X_val], ignore_index=True)
    y = np.concatenate([dados.y_treino, dados.y_val])
    test_fold = np.concatenate([
        np.full(len(dados.y_treino), -1),
        np.zeros(len(dados.y_val), dtype=int),
    ])
    return X, y, test_fold

"""
As duas abordagens de pré-processamento exigidas pelo enunciado (item 3).

  bruta    — a entrada do modelo é apenas o tabuleiro atual, com as casas
             convertidas em valores numéricos (0 = vazio, 1 = O, 2 = X).

  derivada — a entrada do modelo é um conjunto de features extraídas do
             tabuleiro: quantidade de X, quantidade de O, posições ocupadas,
             linhas com 2 X, linhas com 2 O, casas vazias e jogador da vez.

Manter as duas aqui garante que a comparação entre elas seja honesta: a única
coisa que muda é a representação da entrada.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ttt.config import JOGADOR_O, JOGADOR_X, VAZIO
from ttt.game_rules import jogador_da_vez, linhas_com_duas

ABORDAGENS = {
    "bruta": "Tabuleiro atual em valores numéricos, codificado em one-hot (27 colunas)",
    "derivada": "Features extraídas do tabuleiro, padronizadas (15 colunas)",
}

# Nomes na ordem em que `_features_do_tabuleiro` os produz. Úteis para
# inspecionar importância de features na árvore de decisão e no XGBoost.
NOMES_FEATURES_DERIVADAS = (
    ["qtd_x", "qtd_o", "casas_vazias", "linhas_2x", "linhas_2o", "jogador_da_vez"]
    + [f"ocupada_{i}" for i in range(9)]
)


def _features_do_tabuleiro(board):
    """
    Transforma um tabuleiro em features de mais alto nível.

    "Posições ocupadas" vira uma máscara binária de 9 casas, e não uma
    contagem: a contagem já é dada por `casas_vazias`, então é a máscara que
    acrescenta informação sobre a geometria do tabuleiro.
    """
    return [
        board.count(JOGADOR_X),
        board.count(JOGADOR_O),
        board.count(VAZIO),
        linhas_com_duas(board, JOGADOR_X),
        linhas_com_duas(board, JOGADOR_O),
        jogador_da_vez(board),
    ] + [int(casa != VAZIO) for casa in board]


def extrair_features(X):
    """Aplica `_features_do_tabuleiro` a cada linha de X."""
    valores = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    return np.array([_features_do_tabuleiro(list(map(int, linha))) for linha in valores])


def criar_pipeline(abordagem, modelo):
    """
    Monta o pré-processamento da abordagem escolhida com o modelo no final.

    O modelo entra dentro do Pipeline para que o pré-processamento seja
    reajustado a cada fold do GridSearchCV e nunca enxergue os dados de
    validação antes da hora.
    """
    if abordagem == "bruta":
        # One-hot nas 9 casas: cada posição vira 3 colunas (vazio/O/X). Evita
        # que o modelo leia os códigos 0/1/2 como uma escala ordenada — "X" não
        # é o dobro de "O".
        etapas = [("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))]

    elif abordagem == "derivada":
        # As features têm escalas diferentes entre si (contagens de 0 a 9 ao
        # lado de flags 0/1), o que importa para algoritmos sensíveis a
        # distância como k-NN, SVM e MLP.
        etapas = [
            ("features", FunctionTransformer(extrair_features)),
            ("escala", StandardScaler()),
        ]

    else:
        raise ValueError(f"Abordagem '{abordagem}'. Disponíveis: {sorted(ABORDAGENS)}")

    return Pipeline(etapas + [("modelo", modelo)])

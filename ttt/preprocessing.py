"""
As duas abordagens de pré-processamento do item 3 do enunciado.

  bruta    — entrada do modelo é o tabuleiro atual em valores numéricos.
  derivada — entrada do modelo são features extraídas do tabuleiro.

As duas ficam aqui para que a comparação seja honesta: a única coisa que muda
entre elas é a representação da entrada.
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


def _features_do_tabuleiro(board):
    """
    Extrai as features do item 3 de um tabuleiro.

    "Posições ocupadas" é uma máscara binária de 9 casas, e não uma contagem:
    a contagem já vem de `casas_vazias`, então é a máscara que acrescenta
    informação sobre a geometria do tabuleiro.
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
    reajustado a cada fold do GridSearchCV e não enxergue a validação antes da
    hora.
    """
    if abordagem == "bruta":
        # Cada casa vira 3 colunas (vazio/O/X), para o modelo não ler os
        # códigos 0/1/2 como escala ordenada: "X" não é o dobro de "O".
        etapas = [("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))]

    elif abordagem == "derivada":
        # Contagens de 0 a 9 convivem com flags 0/1, e essa diferença de escala
        # distorce k-NN, SVM e MLP, que dependem de distância.
        etapas = [
            ("features", FunctionTransformer(extrair_features)),
            ("escala", StandardScaler()),
        ]

    else:
        raise ValueError(f"Abordagem '{abordagem}'. Disponíveis: {sorted(ABORDAGENS)}")

    return Pipeline(etapas + [("modelo", modelo)])

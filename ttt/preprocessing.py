"""
As duas abordagens de pré-processamento exigidas pelo enunciado (item 3).

  bruta    — a entrada do modelo é apenas o tabuleiro atual, com as casas
             convertidas em valores numéricos (0 = vazio, 1 = O, 2 = X).

  derivada — a entrada do modelo é um conjunto de features extraídas do
             tabuleiro: quantidade de X, quantidade de O, posições ocupadas,
             linhas com 2 X, linhas com 2 O, casas vazias e jogador da vez.

Cada abordagem é uma `Abordagem`, que expõe um `Pipeline` do sklearn. Assim
todo algoritmo recebe as duas do mesmo jeito e a comparação entre elas fica
honesta: a única coisa que muda é a representação da entrada.

Manter isso em um único módulo também elimina o risco que existia antes, em que
cada script aplicava um encoding diferente (um usava one-hot, outro
StandardScaler sobre os códigos brutos) e os resultados não eram comparáveis.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ttt.config import JOGADOR_O, JOGADOR_X, VAZIO
from ttt.game_rules import jogador_da_vez, linhas_com_duas

# --------------------------------------------------------------------------
# Abordagem 2 — extração de features
# --------------------------------------------------------------------------

# Nomes na mesma ordem em que `_extrair_features` os produz. Servem para
# inspecionar importância de features (útil na árvore de decisão e no XGBoost).
NOMES_FEATURES_DERIVADAS = (
    ["qtd_x", "qtd_o", "casas_vazias", "linhas_2x", "linhas_2o", "jogador_da_vez"]
    + [f"ocupada_{i}" for i in range(9)]
)


def _extrair_features(board: list[int]) -> list[int]:
    """
    Transforma um tabuleiro em features de mais alto nível.

    "Posições ocupadas" é representado como uma máscara binária de 9 casas
    (quais casas estão ocupadas), e não como uma contagem — a contagem já é
    dada por `casas_vazias`, então a máscara é o que de fato acrescenta
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


class ExtratorDeFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer do sklearn que aplica `_extrair_features` linha a linha.

    Não tem estado aprendido: `fit` só existe para respeitar a interface e
    permitir que o extrator entre em um `Pipeline`.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        valores = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        return np.array([_extrair_features(list(map(int, linha))) for linha in valores])

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(NOMES_FEATURES_DERIVADAS, dtype=object)


# --------------------------------------------------------------------------
# Registro das abordagens
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Abordagem:
    """Uma abordagem de pré-processamento, pronta para entrar num Pipeline."""

    nome: str
    descricao: str
    _construir: Callable[[], list[tuple[str, BaseEstimator]]] = field(repr=False)

    def pipeline(self, modelo: BaseEstimator | None = None) -> Pipeline:
        """
        Monta o pipeline da abordagem. Se `modelo` for informado, ele é
        encaixado no final — o pré-processamento passa então a ser ajustado
        dentro de cada fold do GridSearchCV, sem vazar dados da validação.
        """
        etapas = list(self._construir())
        if modelo is not None:
            etapas.append(("modelo", modelo))
        return Pipeline(etapas)


def _etapas_bruta() -> list[tuple[str, BaseEstimator]]:
    # One-hot nas 9 casas: cada posição vira 3 colunas (vazio/O/X). Evita que o
    # modelo interprete os códigos 0/1/2 como uma escala ordenada, o que não
    # faz sentido — "X" não é o dobro de "O".
    return [
        ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
    ]


def _etapas_derivada() -> list[tuple[str, BaseEstimator]]:
    # As features derivadas têm escalas diferentes entre si (contagens de 0 a 9
    # convivendo com flags 0/1), então a padronização importa para algoritmos
    # sensíveis a distância, como k-NN, SVM e MLP.
    return [
        ("features", ExtratorDeFeatures()),
        ("escala", StandardScaler()),
    ]


ABORDAGENS: dict[str, Abordagem] = {
    "bruta": Abordagem(
        nome="bruta",
        descricao="Tabuleiro atual em valores numéricos, codificado em one-hot (27 colunas)",
        _construir=_etapas_bruta,
    ),
    "derivada": Abordagem(
        nome="derivada",
        descricao="Features extraídas do tabuleiro, padronizadas (15 colunas)",
        _construir=_etapas_derivada,
    ),
}


def obter(nome: str) -> Abordagem:
    """Busca uma abordagem pelo nome, com erro explícito se não existir."""
    if nome not in ABORDAGENS:
        raise KeyError(f"Abordagem '{nome}'. Disponíveis: {sorted(ABORDAGENS)}")
    return ABORDAGENS[nome]

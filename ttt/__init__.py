"""
Pacote compartilhado do T1 — Tic Tac Toe com ML.

Centraliza tudo o que antes estava duplicado (e divergente) entre os scripts
de cada algoritmo: caminhos, nomes de classes, carga dos splits,
pré-processamento e avaliação.
"""

from ttt.config import (
    CLASS_MAP,
    CLASS_NAMES,
    ENCODE_CASA,
    FEATURE_COLS,
    RANDOM_STATE,
    TARGET_COL,
)

__all__ = [
    "CLASS_MAP",
    "CLASS_NAMES",
    "ENCODE_CASA",
    "FEATURE_COLS",
    "RANDOM_STATE",
    "TARGET_COL",
]

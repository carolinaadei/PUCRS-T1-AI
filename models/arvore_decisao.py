"""
Algoritmo 2 — Árvore de Decisão (exigido pelo enunciado).

Execução:
    python models/arvore_decisao.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.tree import DecisionTreeClassifier

from ttt.config import RANDOM_STATE
from ttt.experiment import cli, executar_cli

NOME = "Arvore de Decisao"

PARAM_GRID = {
    "max_depth": [3, 5, 7, 10, None],
    "criterion": ["gini", "entropy"],
    "min_samples_leaf": [1, 3, 5],
}


def criar_modelo() -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=RANDOM_STATE)


if __name__ == "__main__":
    executar_cli(NOME, criar_modelo, PARAM_GRID, cli(__doc__))

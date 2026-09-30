"""
Algoritmo 2 — Árvore de Decisão (exigido pelo enunciado).

Execução:
    python models/arvore_decisao.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.tree import DecisionTreeClassifier

from ttt import experiment
from ttt.config import RANDOM_STATE

NOME = "Arvore de Decisao"
MODELO = DecisionTreeClassifier(random_state=RANDOM_STATE)

PARAM_GRID = {
    "max_depth": [3, 5, 7, 10, None],
    "criterion": ["gini", "entropy"],
    "min_samples_leaf": [1, 3, 5],
}

if __name__ == "__main__":
    experiment.main(NOME, MODELO, PARAM_GRID)

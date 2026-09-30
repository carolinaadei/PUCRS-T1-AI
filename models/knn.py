"""
Algoritmo 1 — k-Nearest Neighbors (exigido pelo enunciado).

Execução:
    python models/knn.py                      # roda as duas abordagens
    python models/knn.py --abordagem bruta
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.neighbors import KNeighborsClassifier

from ttt.experiment import cli, executar_cli

NOME = "k-NN"

# k ímpar evita empates na votação entre as 5 classes.
PARAM_GRID = {
    "n_neighbors": [1, 3, 5, 7, 9, 11, 15, 19],
    "metric": ["euclidean", "manhattan"],
    "weights": ["uniform", "distance"],
}


def criar_modelo() -> KNeighborsClassifier:
    return KNeighborsClassifier()


if __name__ == "__main__":
    executar_cli(NOME, criar_modelo, PARAM_GRID, cli(__doc__))

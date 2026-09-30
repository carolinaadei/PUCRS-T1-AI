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

from ttt import experiment

NOME = "k-NN"
MODELO = KNeighborsClassifier()

# k ímpar evita empates na votação entre as 5 classes.
PARAM_GRID = {
    "n_neighbors": [1, 3, 5, 7, 9, 11, 15, 19],
    "metric": ["euclidean", "manhattan"],
    "weights": ["uniform", "distance"],
}

if __name__ == "__main__":
    experiment.main(NOME, MODELO, PARAM_GRID)

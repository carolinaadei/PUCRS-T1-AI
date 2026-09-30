"""
SVM — algoritmo de escolha livre do grupo. Ver `reports/docs/svm.md`.

    python models/svm.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.svm import SVC

from ttt import experiment
from ttt.config import RANDOM_STATE

NOME = "SVM"
MODELO = SVC(random_state=RANDOM_STATE)

# O kernel entra na grade junto com C e gamma, para ser escolhido pelo conjunto
# de validação em vez de fixado à mão.
PARAM_GRID = {
    "kernel": ["linear", "rbf", "poly"],
    "C": [0.1, 1, 10, 50, 100],
    "gamma": ["scale", 0.01, 0.1, 1],
}

if __name__ == "__main__":
    experiment.main(NOME, MODELO, PARAM_GRID)

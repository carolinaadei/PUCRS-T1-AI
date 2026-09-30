"""
Algoritmo 4 — SVM (escolha livre do grupo).

Explicação do funcionamento em `reports/docs/svm.md`.

Execução:
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

# O kernel entra na grade junto com C e gamma: assim a escolha do kernel também
# é decidida pelo conjunto de validação, e não fixada à mão.
PARAM_GRID = {
    "kernel": ["linear", "rbf", "poly"],
    "C": [0.1, 1, 10, 50, 100],
    "gamma": ["scale", 0.01, 0.1, 1],
}

if __name__ == "__main__":
    experiment.main(NOME, MODELO, PARAM_GRID)

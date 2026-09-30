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

from ttt.config import RANDOM_STATE
from ttt.experiment import cli, executar_cli

NOME = "SVM"

# O kernel entra na grade junto com C e gamma: assim a escolha do kernel também
# é decidida pelo conjunto de validação, e não fixada à mão como antes.
PARAM_GRID = {
    "kernel": ["linear", "rbf", "poly"],
    "C": [0.1, 1, 10, 50, 100],
    "gamma": ["scale", 0.01, 0.1, 1],
}


def criar_modelo() -> SVC:
    return SVC(random_state=RANDOM_STATE)


if __name__ == "__main__":
    executar_cli(NOME, criar_modelo, PARAM_GRID, cli(__doc__))

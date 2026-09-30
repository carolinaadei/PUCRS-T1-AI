"""
Algoritmo 5 — XGBoost (escolha livre do grupo).

Explicação do funcionamento em `reports/docs/xgboost.md`.

Execução:
    python models/xgboost_clf.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xgboost import XGBClassifier

from ttt.config import RANDOM_STATE
from ttt.experiment import cli, executar_cli

NOME = "XGBoost"

PARAM_GRID = {
    "learning_rate": [0.05, 0.1, 0.2],
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "gamma": [0, 0.1],
}


def criar_modelo() -> XGBClassifier:
    # O problema tem 5 classes. A versão anterior deste script usava
    # objective='binary:logistic', que é para 2 classes — os resultados
    # reportados com aquela configuração não valem.
    # `num_class` não é passado de propósito: o wrapper sklearn do XGBoost o
    # infere dos rótulos, e informá-lo à mão entra em conflito com essa
    # inferência.
    return XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=1,  # o paralelismo fica por conta do GridSearchCV
    )


if __name__ == "__main__":
    executar_cli(NOME, criar_modelo, PARAM_GRID, cli(__doc__))

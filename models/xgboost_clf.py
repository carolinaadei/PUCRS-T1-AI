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

from ttt import experiment
from ttt.config import RANDOM_STATE

NOME = "XGBoost"

# O problema tem 5 classes. A versão anterior usava objective='binary:logistic',
# que é para 2 classes. `num_class` não é informado de propósito: o wrapper
# sklearn o infere dos rótulos, e defini-lo à mão conflita com essa inferência.
MODELO = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=RANDOM_STATE,
    n_jobs=1,  # o paralelismo fica por conta do GridSearchCV
)

PARAM_GRID = {
    "learning_rate": [0.05, 0.1, 0.2],
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "gamma": [0, 0.1],
}

if __name__ == "__main__":
    experiment.main(NOME, MODELO, PARAM_GRID)

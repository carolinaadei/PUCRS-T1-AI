"""
XGBoost — algoritmo de escolha livre do grupo. Ver `reports/docs/xgboost.md`.

    python models/xgboost_clf.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xgboost import XGBClassifier

from ttt import experiment
from ttt.config import RANDOM_STATE

NOME = "XGBoost"

# `num_class` é omitido de propósito: o wrapper sklearn o infere dos rótulos, e
# defini-lo à mão conflita com essa inferência.
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

"""
Algoritmo 3 — Multi-Layer Perceptron (exigido pelo enunciado).

Nota sobre o alvo: ele permanece numérico durante todo o treinamento. O
`early_stopping` interno do MLPClassifier é incompatível com rótulos em texto
(tentaria aplicar np.isnan sobre strings). A tradução para nomes legíveis
acontece só na avaliação, via `ttt.config.CLASS_MAP`.

Execução:
    python models/mlp.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.neural_network import MLPClassifier

from ttt import experiment
from ttt.config import RANDOM_STATE

NOME = "MLP"

# early_stopping separa internamente uma fatia do treino para interromper o
# ajuste quando a perda para de melhorar — principal defesa contra overfitting
# num dataset pequeno como este.
MODELO = MLPClassifier(
    max_iter=2000,
    early_stopping=True,
    n_iter_no_change=20,
    random_state=RANDOM_STATE,
)

PARAM_GRID = {
    "hidden_layer_sizes": [(10,), (50,), (10, 10), (50, 25)],
    "activation": ["relu", "tanh"],
    "learning_rate_init": [0.001, 0.01],
}

if __name__ == "__main__":
    experiment.main(NOME, MODELO, PARAM_GRID)

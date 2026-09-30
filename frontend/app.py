from flask import Flask, jsonify, request, render_template
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import numpy as np
import random
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURE_COLS = ["tl", "tm", "tr", "ml", "mm", "mr", "bl", "bm", "br"]
TARGET_COL = "classe_id"

treino = pd.read_csv(os.path.join(BASE_DIR, 'treino.csv'))
validacao = pd.read_csv(os.path.join(BASE_DIR, 'validacao.csv'))

X_treino = treino[FEATURE_COLS]
y_treino = treino[TARGET_COL].to_numpy()

X_val = validacao[FEATURE_COLS]
y_val = validacao[TARGET_COL].to_numpy()

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_treino_enc = encoder.fit_transform(X_treino)
X_val_enc = encoder.transform(X_val)

X_treino_val = np.vstack([X_treino_enc, X_val_enc])
y_treino_val = np.concatenate([y_treino, y_val])

test_fold = np.concatenate([
    np.full(len(X_treino_enc), -1, dtype=int),
    np.zeros(len(X_val_enc), dtype=int),
])

PARAM_GRID = {
    "hidden_layer_sizes": [(10,), (50,), (10, 10), (50, 25)],
    "activation": ["relu", "tanh"],
    "learning_rate_init": [0.001, 0.01],
}

mlp_base = MLPClassifier(
    max_iter=2000,
    early_stopping=True,
    n_iter_no_change=20,
    random_state=42,
)

print("[INFO] Treinando modelo MLP com GridSearchCV...", flush=True)
grid_search = GridSearchCV(
    estimator=mlp_base,
    param_grid=PARAM_GRID,
    cv=PredefinedSplit(test_fold),
    scoring='accuracy',
    n_jobs=1,
    verbose=0,
)
grid_search.fit(X_treino_val, y_treino_val)
model = grid_search.best_estimator_

CLASS_NAMES = {
    0: 'Empate',
    1: 'X Vence',
    2: 'Possibilidade de Fim de Jogo',
    3: 'Tem Jogo',
    4: 'O Vence'
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/classify', methods=['POST'])
def classify():
    data = request.get_json()
    board = data['board']
    X = pd.DataFrame([board], columns=FEATURE_COLS)
    X_enc = encoder.transform(X)
    pred = int(model.predict(X_enc)[0])
    return jsonify({'class_id': pred, 'class_name': CLASS_NAMES.get(pred, f'Classe {pred}')})


@app.route('/api/computer_move', methods=['POST'])
def computer_move():
    data = request.get_json()
    board = data['board']
    empty = [i for i, v in enumerate(board) if v == 0]
    move = random.choice(empty) if empty else -1
    return jsonify({'move': move})


if __name__ == '__main__':
    print(f"[INFO] Modelo MLP treinado. Melhores params: {grid_search.best_params_}. Iniciando servidor...", flush=True)
    app.run(debug=True, port=5001)

"""
Front end para interagir com o classificador (item 6 do enunciado).

O humano joga de X contra um computador que joga de O aleatoriamente. A cada
jogada o tabuleiro é enviado ao classificador, e a tela compara a predição com
o estado real para contabilizar a acurácia da IA.

O modelo é carregado de `artifacts/`, então é preciso treinar antes:

    python models/comparar.py
    python frontend/app.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from ttt import evaluation, game_rules
from ttt.config import ARTIFACTS_DIR, CLASS_MAP, FEATURE_COLS, VAZIO

app = Flask(__name__)


def carregar_melhor_modelo():
    """
    Carrega o pipeline do algoritmo com melhor acurácia de teste registrada.

    Sem resultados registrados, usa qualquer artefato disponível, para o front
    end subir mesmo antes de a comparação completa ter sido rodada.
    """
    resultados = evaluation.carregar_resultados()

    if not resultados.empty:
        for _, linha in resultados.sort_values("acuracia_teste", ascending=False).iterrows():
            nome = f"{linha['algoritmo'].lower().replace(' ', '_')}_{linha['abordagem']}.joblib"
            if (ARTIFACTS_DIR / nome).exists():
                return joblib.load(ARTIFACTS_DIR / nome), Path(nome).stem

    disponiveis = sorted(ARTIFACTS_DIR.glob("*.joblib"))
    if not disponiveis:
        raise FileNotFoundError(
            "Nenhum modelo treinado em artifacts/.\n"
            "Rode primeiro:  python models/comparar.py"
        )
    return joblib.load(disponiveis[0]), disponiveis[0].stem


MODELO, MODELO_NOME = carregar_melhor_modelo()
print(f"[INFO] Modelo carregado: {MODELO_NOME}", flush=True)


@app.route("/")
def index():
    return render_template("index.html", modelo_nome=MODELO_NOME)


@app.route("/api/classify", methods=["POST"])
def classify():
    """
    Predição da IA para o tabuleiro recebido, junto do estado real.

    O estado real vai na resposta para que o front end não precise ter a sua
    própria cópia das regras nem dos rótulos das classes.
    """
    board = _ler_tabuleiro(request.get_json())

    predito = int(MODELO.predict(pd.DataFrame([board], columns=FEATURE_COLS))[0])
    real = game_rules.classificar_id(board)

    return jsonify({
        "class_id": predito,
        "class_name": CLASS_MAP.get(predito, f"Classe {predito}"),
        "real_id": real,
        "real_name": CLASS_MAP[real],
        "correto": predito == real,
        "fim_de_jogo": game_rules.fim_de_jogo(board),
    })


@app.route("/api/computer_move", methods=["POST"])
def computer_move():
    """Jogada do computador (O), escolhida aleatoriamente entre as casas vazias."""
    board = _ler_tabuleiro(request.get_json())
    vazias = [i for i, casa in enumerate(board) if casa == VAZIO]

    return jsonify({"move": random.choice(vazias) if vazias else -1})


def _ler_tabuleiro(payload):
    """Valida o tabuleiro recebido: 9 casas com valores 0, 1 ou 2."""
    board = (payload or {}).get("board")

    if not isinstance(board, list) or len(board) != 9:
        raise ValueError("'board' deve ser uma lista de 9 posições")
    if any(casa not in (0, 1, 2) for casa in board):
        raise ValueError("Casas devem ser 0 (vazio), 1 (O) ou 2 (X)")

    return [int(casa) for casa in board]


@app.errorhandler(ValueError)
def _erro_de_entrada(erro):
    return jsonify({"erro": str(erro)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5001)

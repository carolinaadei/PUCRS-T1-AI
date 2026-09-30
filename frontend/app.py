"""
Item 6 do enunciado — front end mínimo para interagir com o classificador.

Um humano joga de X contra um computador que joga de O aleatoriamente. A cada
jogada o tabuleiro é enviado ao classificador, que informa o estado do jogo; o
front end compara essa predição com o estado real e contabiliza a acurácia da
IA durante a interação.

O servidor **carrega** o melhor modelo já treinado em `artifacts/`, em vez de
treinar na inicialização como antes. Treinar a cada boot deixava a subida lenta
e, pior, fazia o front end usar um modelo diferente do que foi avaliado no
relatório.

Execução:
    python models/comparar.py     # treina e escolhe o melhor (uma vez)
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
from ttt.config import (
    ARTIFACTS_DIR,
    CLASS_MAP,
    FEATURE_COLS,
    JOGADOR_O,
    VAZIO,
)

app = Flask(__name__)


# --------------------------------------------------------------------------
# Carregamento do modelo
# --------------------------------------------------------------------------


def _melhor_artefato() -> Path | None:
    """
    Escolhe o artefato do algoritmo com melhor acurácia de teste registrada.

    Se nada foi registrado ainda, cai para qualquer .joblib disponível — assim
    o front end sobe mesmo antes de a comparação completa ter sido rodada.
    """
    resultados = evaluation.carregar_resultados()

    if not resultados.empty:
        for _, linha in resultados.sort_values("acuracia_teste", ascending=False).iterrows():
            nome = f"{linha['algoritmo'].lower().replace(' ', '_')}_{linha['abordagem']}.joblib"
            caminho = ARTIFACTS_DIR / nome
            if caminho.exists():
                return caminho

    disponiveis = sorted(ARTIFACTS_DIR.glob("*.joblib"))
    return disponiveis[0] if disponiveis else None


def carregar_modelo():
    """Carrega o pipeline treinado (pré-processamento + classificador)."""
    caminho = _melhor_artefato()
    if caminho is None:
        raise FileNotFoundError(
            "Nenhum modelo treinado em artifacts/.\n"
            "Rode primeiro:  python models/comparar.py"
        )

    print(f"[INFO] Modelo carregado: {caminho.name}", flush=True)
    return joblib.load(caminho), caminho.stem


MODELO, MODELO_NOME = carregar_modelo()


# --------------------------------------------------------------------------
# Rotas
# --------------------------------------------------------------------------


@app.route("/")
def index():
    # CLASS_MAP vai para o template para que o front end nunca tenha a sua
    # própria cópia dos rótulos — era exatamente essa duplicata que estava
    # divergindo e invertendo "X vence" com "O vence".
    return render_template(
        "index.html",
        class_map=CLASS_MAP,
        modelo_nome=MODELO_NOME,
    )


@app.route("/api/classify", methods=["POST"])
def classify():
    """Predição da IA para o tabuleiro recebido."""
    board = _ler_tabuleiro(request.get_json())

    X = pd.DataFrame([board], columns=FEATURE_COLS)
    pred = int(MODELO.predict(X)[0])

    # Estado real calculado pelas regras do jogo, para o front end contabilizar
    # o acerto sem precisar reimplementar a lógica.
    real = game_rules.classificar_id(board)

    return jsonify({
        "class_id": pred,
        "class_name": CLASS_MAP.get(pred, f"Classe {pred}"),
        "real_id": real,
        "real_name": CLASS_MAP[real],
        "correto": pred == real,
        "fim_de_jogo": game_rules.fim_de_jogo(board),
    })


@app.route("/api/computer_move", methods=["POST"])
def computer_move():
    """Jogada do computador (O), escolhida aleatoriamente entre as casas vazias."""
    board = _ler_tabuleiro(request.get_json())
    vazias = [i for i, v in enumerate(board) if v == VAZIO]

    return jsonify({"move": random.choice(vazias) if vazias else -1, "jogador": JOGADOR_O})


@app.route("/api/status")
def status():
    """Informa qual modelo está servindo as predições."""
    return jsonify({"modelo": MODELO_NOME, "classes": CLASS_MAP})


def _ler_tabuleiro(payload: dict | None) -> list[int]:
    """Valida o tabuleiro recebido: 9 casas com valores 0, 1 ou 2."""
    board = (payload or {}).get("board")

    if not isinstance(board, list) or len(board) != 9:
        raise ValueError("'board' deve ser uma lista de 9 posições")
    if any(v not in (0, 1, 2) for v in board):
        raise ValueError("Casas devem ser 0 (vazio), 1 (O) ou 2 (X)")

    return [int(v) for v in board]


@app.errorhandler(ValueError)
def _erro_de_entrada(erro: ValueError):
    return jsonify({"erro": str(erro)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5001)

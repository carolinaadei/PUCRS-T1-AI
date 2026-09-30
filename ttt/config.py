"""
Constantes e caminhos do projeto.

Este é o único lugar onde o mapeamento de classes é declarado. Antes cada
script tinha a sua própria cópia e elas divergiam entre si (o front end trocava
"O vence" com "X vence"), o que corrompia a contagem de acertos da IA.

O mapeamento foi validado contra os CSVs: ele corresponde à ordem alfabética
produzida pelo LabelEncoder em `notebooks/01_construcao_dataset.ipynb`.
"""

from pathlib import Path

# Caminhos sempre derivados da raiz do repositório, nunca do diretório atual
ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_UCI_FILE = ROOT_DIR / "data" / "raw" / "tic-tac-toe.data"
TREINO_CSV = ROOT_DIR / "data" / "processed" / "treino.csv"
VALIDACAO_CSV = ROOT_DIR / "data" / "processed" / "validacao.csv"
TESTE_CSV = ROOT_DIR / "data" / "processed" / "teste.csv"

FIGURES_DIR = ROOT_DIR / "reports" / "figures"
METRICS_DIR = ROOT_DIR / "reports" / "metrics"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

for pasta in (FIGURES_DIR, METRICS_DIR, ARTIFACTS_DIR):
    pasta.mkdir(parents=True, exist_ok=True)

# Posições do tabuleiro 3x3: top/middle/bottom x left/middle/right
FEATURE_COLS = ["tl", "tm", "tr", "ml", "mm", "mr", "bl", "bm", "br"]
TARGET_COL = "classe_id"

# Codificação das casas, usada nos CSVs e no front end
VAZIO, JOGADOR_O, JOGADOR_X = 0, 1, 2

CLASS_MAP = {
    0: "Empate",
    1: "O vence",
    2: "Possibilidade de Fim de Jogo",
    3: "Tem jogo",
    4: "X vence",
}

# Lista ordenada por id — formato esperado por `target_names` do sklearn
CLASS_NAMES = [CLASS_MAP[i] for i in sorted(CLASS_MAP)]

RANDOM_STATE = 42

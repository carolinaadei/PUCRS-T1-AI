"""
Constantes e caminhos do projeto.

Este é o único lugar onde o mapeamento de classes é declarado. Antes da
reorganização cada script tinha a sua própria cópia e elas divergiam entre si
(o front end trocava "O vence" com "X vence"), o que corrompia a contabilização
de acertos da IA durante a interação.

O mapeamento abaixo foi validado contra os CSVs: ele corresponde à ordem
alfabética produzida pelo LabelEncoder em `notebooks/01_construcao_dataset.ipynb`.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Caminhos — sempre derivados da raiz do repositório, nunca do diretório atual
# --------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

RAW_UCI_FILE = RAW_DIR / "tic-tac-toe.data"
TREINO_CSV = PROCESSED_DIR / "treino.csv"
VALIDACAO_CSV = PROCESSED_DIR / "validacao.csv"
TESTE_CSV = PROCESSED_DIR / "teste.csv"

# --------------------------------------------------------------------------
# Esquema dos dados
# --------------------------------------------------------------------------

# Posições do tabuleiro 3x3: top/middle/bottom x left/middle/right
FEATURE_COLS = ["tl", "tm", "tr", "ml", "mm", "mr", "bl", "bm", "br"]
TARGET_COL = "classe_id"

# Codificação das casas do tabuleiro (usada nos CSVs processados)
ENCODE_CASA = {"b": 0, "o": 1, "x": 2}
VAZIO, JOGADOR_O, JOGADOR_X = 0, 1, 2

# Rótulo legível de cada classe. Ordem = LabelEncoder (alfabética).
CLASS_MAP = {
    0: "Empate",
    1: "O vence",
    2: "Possibilidade de Fim de Jogo",
    3: "Tem jogo",
    4: "X vence",
}

# Lista ordenada por id — formato esperado por `target_names` do sklearn
CLASS_NAMES = [CLASS_MAP[i] for i in sorted(CLASS_MAP)]

# --------------------------------------------------------------------------
# Reprodutibilidade
# --------------------------------------------------------------------------

RANDOM_STATE = 42


def garantir_diretorios() -> None:
    """Cria os diretórios de saída, caso ainda não existam."""
    for d in (FIGURES_DIR, METRICS_DIR, ARTIFACTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

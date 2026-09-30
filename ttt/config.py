"""
Constantes e caminhos do projeto.

O mapeamento de classes é declarado só aqui. A ordem dos ids é a do
LabelEncoder usado em `notebooks/01_construcao_dataset.ipynb`, que ordena os
nomes alfabeticamente.
"""

from pathlib import Path

# Derivado do arquivo, não do diretório de execução: os scripts funcionam
# rodados de qualquer lugar.
ROOT_DIR = Path(__file__).resolve().parent.parent

TREINO_CSV = ROOT_DIR / "data" / "processed" / "treino.csv"
VALIDACAO_CSV = ROOT_DIR / "data" / "processed" / "validacao.csv"
TESTE_CSV = ROOT_DIR / "data" / "processed" / "teste.csv"

FIGURES_DIR = ROOT_DIR / "reports" / "figures"
METRICS_DIR = ROOT_DIR / "reports" / "metrics"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

for pasta in (FIGURES_DIR, METRICS_DIR, ARTIFACTS_DIR):
    pasta.mkdir(parents=True, exist_ok=True)

# Posições do tabuleiro: top/middle/bottom x left/middle/right
FEATURE_COLS = ["tl", "tm", "tr", "ml", "mm", "mr", "bl", "bm", "br"]
TARGET_COL = "classe_id"

VAZIO, JOGADOR_O, JOGADOR_X = 0, 1, 2

CLASS_MAP = {
    0: "Empate",
    1: "O vence",
    2: "Possibilidade de Fim de Jogo",
    3: "Tem jogo",
    4: "X vence",
}

# Ordenado por id: formato que o sklearn espera em `target_names`
CLASS_NAMES = [CLASS_MAP[i] for i in sorted(CLASS_MAP)]

RANDOM_STATE = 42

"""
Regras do jogo da velha — a "verdade" sobre o estado de um tabuleiro.

Esta lógica era definida apenas dentro do notebook de construção do dataset,
mas é necessária em três lugares: para rotular o dataset, para derivar as
features da 2ª abordagem de pré-processamento e para que o front end saiba o
estado real e possa contabilizar os acertos da IA.

Os tabuleiros são listas de 9 inteiros na codificação de `config.ENCODE_CASA`
(0 = vazio, 1 = O, 2 = X), lidos da esquerda para a direita e de cima para
baixo.
"""

from ttt.config import CLASS_MAP, ENCODE_CASA, JOGADOR_O, JOGADOR_X, VAZIO

Tabuleiro = list[int]

# Índices das 8 combinações que fecham três em linha
LINHAS_VENCEDORAS = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],   # linhas
    [0, 3, 6], [1, 4, 7], [2, 5, 8],   # colunas
    [0, 4, 8], [2, 4, 6],              # diagonais
]

# Rótulo -> id, para quem já tem o nome da classe e quer o alvo numérico
CLASSE_PARA_ID = {nome: idx for idx, nome in CLASS_MAP.items()}


def de_simbolos(board: list[str]) -> Tabuleiro:
    """Converte um tabuleiro no formato original do UCI ('x'/'o'/'b') para inteiros."""
    return [ENCODE_CASA[c] for c in board]


def vencedor(board: Tabuleiro) -> int | None:
    """Retorna o jogador que fechou três em linha, ou None se não houver."""
    for linha in LINHAS_VENCEDORAS:
        a, b, c = (board[i] for i in linha)
        if a == b == c and a != VAZIO:
            return a
    return None


def linhas_com_duas(board: Tabuleiro, jogador: int) -> int:
    """
    Conta as linhas em que `jogador` tem duas marcas e a terceira casa está
    vazia — ou seja, ameaças de vitória imediata.
    """
    total = 0
    for linha in LINHAS_VENCEDORAS:
        vals = [board[i] for i in linha]
        if vals.count(jogador) == 2 and vals.count(VAZIO) == 1:
            total += 1
    return total


def pode_vencer_prox(board: Tabuleiro, jogador: int) -> bool:
    """Indica se `jogador` vence na próxima jogada."""
    return linhas_com_duas(board, jogador) > 0


def jogador_da_vez(board: Tabuleiro) -> int:
    """
    Determina de quem é a vez. X sempre abre a partida, então joga quando o
    número de marcas de ambos é igual.
    """
    return JOGADOR_X if board.count(JOGADOR_X) == board.count(JOGADOR_O) else JOGADOR_O


def classificar(board: Tabuleiro) -> str:
    """
    Classifica o tabuleiro em uma das cinco classes do trabalho.

    A ordem dos testes importa: uma vitória encerra a partida e tem prioridade
    sobre "possibilidade de fim de jogo".
    """
    w = vencedor(board)
    if w == JOGADOR_X:
        return "X vence"
    if w == JOGADOR_O:
        return "O vence"
    if board.count(VAZIO) == 0:
        return "Empate"
    if pode_vencer_prox(board, JOGADOR_X) or pode_vencer_prox(board, JOGADOR_O):
        return "Possibilidade de Fim de Jogo"
    return "Tem jogo"


def classificar_id(board: Tabuleiro) -> int:
    """Mesma classificação de `classificar`, porém devolvendo o id numérico."""
    return CLASSE_PARA_ID[classificar(board)]


def fim_de_jogo(board: Tabuleiro) -> bool:
    """Indica se a partida acabou (alguém venceu ou o tabuleiro encheu)."""
    return vencedor(board) is not None or board.count(VAZIO) == 0


def tabuleiro_valido(board: Tabuleiro) -> bool:
    """
    Verifica se o estado é alcançável em uma partida real: alternância correta
    de jogadas e no máximo um vencedor.
    """
    cx, co = board.count(JOGADOR_X), board.count(JOGADOR_O)
    if cx not in (co, co + 1):
        return False

    w = vencedor(board)
    if w == JOGADOR_X and cx != co + 1:
        return False
    if w == JOGADOR_O and cx != co:
        return False
    return True

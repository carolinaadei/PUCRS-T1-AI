"""
Regras do jogo da velha: o estado real de um tabuleiro.

Usado para rotular o dataset, derivar as features da abordagem 2 e informar o
estado real ao front end.

Tabuleiro = lista de 9 inteiros (0 vazio, 1 O, 2 X), da esquerda para a direita
e de cima para baixo.
"""

from ttt.config import CLASS_MAP, JOGADOR_O, JOGADOR_X, VAZIO

# As 8 combinações que fecham três em linha
LINHAS_VENCEDORAS = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],   # linhas
    [0, 3, 6], [1, 4, 7], [2, 5, 8],   # colunas
    [0, 4, 8], [2, 4, 6],              # diagonais
]

_CLASSE_PARA_ID = {nome: idx for idx, nome in CLASS_MAP.items()}


def vencedor(board):
    """Jogador que fechou três em linha, ou None."""
    for linha in LINHAS_VENCEDORAS:
        a, b, c = (board[i] for i in linha)
        if a == b == c and a != VAZIO:
            return a
    return None


def linhas_com_duas(board, jogador):
    """Quantas ameaças de vitória o jogador tem: duas marcas e a casa restante vazia."""
    total = 0
    for linha in LINHAS_VENCEDORAS:
        vals = [board[i] for i in linha]
        if vals.count(jogador) == 2 and vals.count(VAZIO) == 1:
            total += 1
    return total


def jogador_da_vez(board):
    """X abre a partida, então joga quando ambos têm o mesmo número de marcas."""
    return JOGADOR_X if board.count(JOGADOR_X) == board.count(JOGADOR_O) else JOGADOR_O


def classificar(board):
    """
    Classifica o tabuleiro em uma das cinco classes do trabalho.

    A ordem dos testes importa: uma vitória encerra a partida, então tem
    prioridade sobre "possibilidade de fim de jogo".
    """
    w = vencedor(board)
    if w == JOGADOR_X:
        return "X vence"
    if w == JOGADOR_O:
        return "O vence"
    if board.count(VAZIO) == 0:
        return "Empate"
    if linhas_com_duas(board, JOGADOR_X) or linhas_com_duas(board, JOGADOR_O):
        return "Possibilidade de Fim de Jogo"
    return "Tem jogo"


def classificar_id(board):
    """Como `classificar`, mas devolvendo o id numérico."""
    return _CLASSE_PARA_ID[classificar(board)]


def fim_de_jogo(board):
    """A partida acabou: alguém venceu ou o tabuleiro encheu."""
    return vencedor(board) is not None or board.count(VAZIO) == 0

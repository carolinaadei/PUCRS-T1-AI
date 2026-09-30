# 1. Dataset

> Item 2 do enunciado — obter o dataset do UCI, analisar se atende ao problema,
> corrigir o que for preciso e registrar os problemas encontrados e os passos
> executados.

**Código:** [notebooks/01_construcao_dataset.ipynb](../../notebooks/01_construcao_dataset.ipynb) ·
**Regras do jogo:** [ttt/game_rules.py](../../ttt/game_rules.py) ·
**Saída:** [data/processed/](../../data/processed/)

---

## 1.1 O que o UCI entrega

Partimos do [Tic-Tac-Toe Endgame](https://archive.ics.uci.edu/dataset/101/tic+tac+toe+endgame),
em [data/raw/tic-tac-toe.data](../../data/raw/tic-tac-toe.data):

| | |
|---|---|
| Instâncias | 958 |
| Atributos | 9 casas do tabuleiro, valores `x`, `o`, `b` (*blank*) |
| Alvo | `positive` (626) / `negative` (332) |
| Valores nulos | nenhum |

## 1.2 Problemas encontrados

O dataset **não atende ao problema como vem**. São dois problemas, e o segundo é
o mais grave:

**Problema 1 — o alvo tem 2 classes, e o trabalho pede 5.**
`positive`/`negative` significa apenas "X tem três em linha ou não". Não existe
coluna que distinga empate, jogo em andamento ou ameaça de vitória.

**Problema 2 — o dataset só contém posições finais.**
Como o nome diz, é um dataset de *endgame*. Todo tabuleiro nele é uma partida
encerrada. As classes "Tem jogo" e "Possibilidade de Fim de Jogo" descrevem
partidas **em andamento**, e portanto não têm uma única instância no arquivo
original.

Isso ficou evidente ao reclassificar as 958 linhas pelas regras do jogo
(função `classificar` em [ttt/game_rules.py](../../ttt/game_rules.py)):

| Alvo UCI | Empate | O vence | X vence | Tem jogo | Possibilidade |
|---|---:|---:|---:|---:|---:|
| `negative` | 16 | 316 | 0 | 0 | 0 |
| `positive` | 0 | 0 | 626 | 0 | 0 |

Duas leituras importantes desse cruzamento:

- `positive` corresponde **exatamente** a "X vence", nas 626 instâncias. A
  reclassificação não contradiz o rótulo original em nenhum caso.
- `negative` se divide em dois casos que o UCI tratava como um só: 316 vitórias
  de O e 16 empates.

## 1.3 Passos executados

### Passo 1 — Reclassificação lógica

Em vez de confiar no rótulo do UCI, cada tabuleiro foi reclassificado pelas
regras do jogo. A função `classificar` aplica os testes nesta ordem, e a ordem
importa: uma vitória encerra a partida, então tem prioridade sobre "possibilidade
de fim de jogo".

| Ordem | Teste | Classe |
|---|---|---|
| 1 | três em linha de X | X vence |
| 2 | três em linha de O | O vence |
| 3 | tabuleiro cheio | Empate |
| 4 | alguém tem duas marcas numa linha com a terceira casa vazia | Possibilidade de Fim de Jogo |
| 5 | nenhum dos anteriores | Tem jogo |

### Passo 2 — Geração de estados sintéticos

Para as duas classes ausentes, simulamos partidas aleatórias e capturamos os
tabuleiros intermediários. A cada jogada o estado é classificado e guardado; a
simulação para quando a partida acaba. Isso gera estados **legítimos**, porque
todos vêm de sequências de jogadas válidas.

### Passo 3 — Deduplicação

Tabuleiros repetidos foram removidos (`drop_duplicates` sobre as 9 casas). Sem
isso, a simulação inflaria as classes comuns com o mesmo estado várias vezes, e
o mesmo tabuleiro poderia cair em treino e teste ao mesmo tempo — vazamento
direto.

### Passo 4 — Balanceamento em ~200 por classe

O enunciado sugere começar com cerca de 200 amostras por classe. Quatro das
cinco chegaram a 200. **Empate ficou em 16** — ver a seção seguinte.

### Passo 5 — Codificação

As casas foram convertidas para números: `b → 0`, `o → 1`, `x → 2`. Os nomes das
classes viraram ids pelo `LabelEncoder`, que ordena alfabeticamente:

| id | Classe |
|---:|---|
| 0 | Empate |
| 1 | O vence |
| 2 | Possibilidade de Fim de Jogo |
| 3 | Tem jogo |
| 4 | X vence |

Esse mapeamento está declarado uma única vez, em
[ttt/config.py](../../ttt/config.py), e é a fonte da verdade para todo o
projeto — scripts, notebooks e front end.

---

## 1.4 Por que "Empate" tem só 16 amostras

Essa é a limitação mais visível do dataset, e a explicação **não** é falha de
amostragem: **16 é o total de empates que existem no jogo da velha.**

Um empate é um tabuleiro cheio sem vencedor. Como X abre a partida, todo
tabuleiro cheio tem 5 X e 4 O, o que dá C(9,5) = 126 tabuleiros possíveis.
Enumerando os 126 e descartando os que têm três em linha, sobram exatamente 16.

```python
from itertools import combinations
from ttt.game_rules import vencedor

empates = [b for posicoes in combinations(range(9), 5)
           if vencedor(b := [2 if i in posicoes else 1 for i in range(9)]) is None]
len(empates)   # -> 16
```

Ou seja: o dataset contém **100% da classe Empate**. Não há amostra a mais para
coletar, gerar ou balancear. O desbalanceamento é uma propriedade do problema,
não um defeito do nosso processo.

### Como isso aparece no espaço de estados

Enumerando os 3⁹ = 19.683 tabuleiros possíveis e ficando só com os alcançáveis
em uma partida real (alternância correta de jogadas, no máximo um vencedor):

| Classe | Estados que existem | No dataset | Cobertura |
|---|---:|---:|---:|
| Empate | 16 | 16 | **100%** |
| O vence | 358 | 200 | 55,9% |
| Tem jogo | 678 | 200 | 29,5% |
| X vence | 662 | 200 | 30,2% |
| Possibilidade de Fim de Jogo | 3.842 | 200 | 5,2% |
| **Total** | **5.556** | **816** | 14,7% |

Isso também antecipa uma dificuldade: "Possibilidade de Fim de Jogo" é a classe
de longe mais numerosa no jogo real (69% dos estados alcançáveis) e a que o
dataset cobre menos. É esperado que seja a mais difícil de aprender — e é
exatamente o que os resultados mostram (ver [04-resultados.md](04-resultados.md)).

### Consequências práticas

1. As médias de precision, recall e F-measure são **`weighted`**, não `macro`.
   Uma média macro daria a Empate, com 4 amostras no teste, o mesmo peso das
   outras quatro classes com 40 cada.
2. As métricas da classe Empate são estatisticamente frágeis: com 4 amostras no
   teste, um único erro muda o recall em 25 pontos.
3. Nenhuma técnica de balanceamento foi aplicada. Oversampling de 16 estados
   para 200 apenas repetiria os mesmos tabuleiros 12 vezes cada, o que ensinaria
   o modelo a decorá-los em vez de generalizar.

---

## 1.5 Divisão final

Divisão estratificada em 60% treino / 20% validação / 20% teste, gravada em
[data/processed/](../../data/processed/). Os três arquivos são fixos e usados
por todos os algoritmos — ver [03-metodologia.md](03-metodologia.md).

| Classe | Treino | Validação | Teste | Total |
|---|---:|---:|---:|---:|
| Empate | 9 | 3 | 4 | 16 |
| O vence | 120 | 40 | 40 | 200 |
| Possibilidade de Fim de Jogo | 120 | 40 | 40 | 200 |
| Tem jogo | 120 | 40 | 40 | 200 |
| X vence | 120 | 40 | 40 | 200 |
| **Total** | **489** | **163** | **164** | **816** |

---

## 1.6 Verificação

As regras em [ttt/game_rules.py](../../ttt/game_rules.py) reproduzem **816 de
816** rótulos gravados nos CSVs — 100%. Isso confirma duas coisas: que os
rótulos estão coerentes com as regras do jogo, e que o mapeamento de ids em
`CLASS_MAP` está correto.

```bash
python -c "
from ttt import dataset, game_rules
d = dataset.carregar()
pares = [(d.X_treino, d.y_treino), (d.X_val, d.y_val), (d.X_teste, d.y_teste)]
ok = sum(game_rules.classificar_id(list(map(int, linha))) == int(alvo)
         for X, y in pares for linha, alvo in zip(X.to_numpy(), y))
print(ok, 'de 816')
"
```

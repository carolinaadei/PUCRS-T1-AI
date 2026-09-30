# k-Nearest Neighbors (k-NN)

> Algoritmo exigido pelo enunciado.

**Código:** [models/knn.py](../../../models/knn.py) ·
**Notebook exploratório:** [notebooks/02_knn_exploratorio.ipynb](../../../notebooks/02_knn_exploratorio.ipynb)

```bash
python models/knn.py                      # roda as duas abordagens
python models/knn.py --abordagem bruta
```

---

## 1. Como funciona

O k-NN não treina, no sentido usual: ele **guarda** o conjunto de treino. Para
classificar um tabuleiro novo, mede a distância dele até todas as amostras
guardadas, pega as `k` mais próximas e devolve a classe mais votada entre elas.

É um algoritmo *baseado em instâncias* (ou *preguiçoso*): todo o custo está na
predição, não no treino.

**Vantagens**
- Não assume nenhuma forma para a fronteira entre as classes.
- Não tem parâmetros a ajustar no treino — só o `k` e a métrica de distância.
- Fácil de explicar: "esse tabuleiro se parece com esses outros".

**Desvantagens**
- A predição fica mais cara conforme o conjunto de treino cresce.
- Sensível à escala das features e à presença de features irrelevantes.
- Vai mal com classes minoritárias: uma classe com poucas amostras raramente
  ganha a votação. Isso pesa neste trabalho, como se vê nos resultados.

---

## 2. Pré-processamento

O algoritmo é rodado nas duas abordagens de
[02-preprocessamento.md](../02-preprocessamento.md). O k-NN é o caso em que a
padronização mais importa, porque a distância euclidiana é literalmente a regra
de decisão: uma feature com amplitude maior domina o cálculo.

---

## 3. Parâmetros testados

```python
PARAM_GRID = {
    "n_neighbors": [1, 3, 5, 7, 9, 11, 15, 19],
    "metric": ["euclidean", "manhattan"],
    "weights": ["uniform", "distance"],
}
```

| Parâmetro | Valores | Por quê |
|---|---|---|
| `n_neighbors` | 1 a 19, só ímpares | valores ímpares reduzem empate na votação. O limite superior vai até 19 porque, com 489 amostras de treino, vizinhanças maiores começam a atravessar a fronteira entre classes |
| `metric` | `euclidean`, `manhattan` | em dados esparsos e binários como o one-hot, a distância de Manhattan às vezes separa melhor que a euclidiana; vale testar as duas |
| `weights` | `uniform`, `distance` | `distance` dá mais peso aos vizinhos mais próximos, o que pode ajudar as classes minoritárias a vencer a votação quando estão perto |

São 8 × 2 × 2 = **32 combinações**, cada uma pontuada no conjunto de validação.

---

## 4. Resultados

### Execução exploratória (notebook)

Registrada em
[notebooks/02_knn_exploratorio.ipynb](../../../notebooks/02_knn_exploratorio.ipynb),
com `StandardScaler` sobre os códigos brutos e escolha de `k` por laço manual
sobre a validação:

| | |
|---|---|
| Melhor `k` | 19 |
| Acurácia na validação | 0,5890 |
| **Acurácia no teste** | **0,5488** |

Por classe, no conjunto de teste:

| Classe | Precision | Recall | F1 | Suporte |
|---|---:|---:|---:|---:|
| Empate | 0,00 | 0,00 | 0,00 | 4 |
| O vence | 0,46 | 0,65 | 0,54 | 40 |
| Possibilidade de Fim de Jogo | 0,44 | 0,35 | 0,39 | 40 |
| Tem jogo | 0,68 | 0,65 | 0,67 | 40 |
| X vence | 0,65 | 0,60 | 0,62 | 40 |
| **Média weighted** | **0,54** | **0,55** | **0,54** | 164 |

### Análise

**O resultado foi fraco, e o motivo é estrutural.** Uma acurácia de 55% num
problema de 5 classes é bem acima do acaso (20%), mas é o pior desempenho entre
os algoritmos do trabalho.

**Empate com F1 = 0,00 é o achado mais revelador:** o modelo **nunca** prediz
essa classe. Com `k = 19` e apenas 9 amostras de empate no treino, é
aritmeticamente impossível que Empate vença uma votação — mesmo que todos os 9
empates estivessem entre os 19 vizinhos mais próximos, ainda seriam minoria.
Isso não é azar de amostragem: é consequência direta de escolher um `k` maior
que o tamanho da classe minoritária.

**"Possibilidade de Fim de Jogo" também vai mal (F1 = 0,39)**, e pelo motivo
descrito em [01-dataset.md](../01-dataset.md): essa classe é definida por uma
condição de contagem (existe linha com duas marcas e a terceira vazia), não por
semelhança visual entre tabuleiros. Dois tabuleiros podem ser vizinhos muito
próximos em distância euclidiana e pertencer a classes diferentes, porque uma
única casa muda a resposta. O k-NN, que decide por proximidade, não tem como
capturar isso na abordagem bruta.

**As classes que ele acerta são as visualmente distintivas:** "Tem jogo"
(F1 = 0,67) e "X vence" (F1 = 0,62), que têm padrões de ocupação mais
característicos.

### Overfitting?

Não há overfitting: a acurácia de validação (0,589) e a de teste (0,549) são
próximas, e ambas são baixas. O problema aqui é o oposto — **underfitting**. O
modelo é simples demais para a estrutura do problema.

### O que a abordagem derivada deve mudar

A hipótese é que a abordagem `derivada` ajude bastante o k-NN, porque entrega
`linhas_2x` e `linhas_2o` prontas: dois tabuleiros com a mesma contagem de
ameaças passam a ficar próximos no espaço de features, o que é exatamente o que
a distância precisa para funcionar aqui. Rodar `python models/knn.py` mede isso.

> Os números acima vêm da execução exploratória, sob o protocolo antigo. Os
> números comparáveis com os demais algoritmos, sob o protocolo unificado, saem
> de `python models/comparar.py` — ver [04-resultados.md](../04-resultados.md).

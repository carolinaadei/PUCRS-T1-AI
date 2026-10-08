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

| Abordagem | Acur. val. | Acur. teste | Precision | Recall | F1 | Treino (s) | Predição (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bruta` | 72,4% | 75,0% | 0,737 | 0,750 | 0,737 | 0,0254 | 41,18 |
| `derivada` | 76,7% | **79,9%** | 0,820 | 0,799 | **0,800** | 0,0066 | 4,19 |

**Melhores hiperparâmetros**

| Abordagem | Configuração |
|---|---|
| `bruta` | `k=9` · euclidean · distance |
| `derivada` | `k=5` · manhattan · uniform |

### Análise

**O k-NN é o pior dos cinco algoritmos** (79,9%, contra 89,0% do XGBoost), mas
melhora bem com a abordagem derivada: **+4,9 pontos** de acurácia e +0,063 de
F1.

O ganho vem quase todo da classe difícil: o F1 de "Possibilidade de Fim de
Jogo" salta de **0,51 para 0,78**. Faz sentido — com `linhas_2x` e `linhas_2o`
explícitas, dois tabuleiros com o mesmo número de ameaças passam a ficar
próximos no espaço de features, que é exatamente o que a distância precisa para
funcionar. Na abordagem bruta eles podiam estar longe um do outro, porque o
one-hot mede diferença de *casas*, não de *ameaças*.

**O `k` caiu de 9 para 5 na derivada.** Vizinhanças menores funcionam melhor
quando as features já separam as classes: não é preciso votar sobre uma região
grande para filtrar ruído.

**A métrica mudou de euclidean para manhattan.** Em features de contagem, a
distância de Manhattan soma as diferenças absolutas — "este tabuleiro tem 2
ameaças a mais e 1 casa vazia a menos" — que é uma noção de semelhança mais
natural aqui do que a euclidiana.

### O custo de predição: o pior do trabalho

**41,18 ms na abordagem bruta** — 17× o segundo colocado, e o único valor do
trabalho que chega a ser perceptível.

A causa é estrutural. O k-NN não tem modelo: ele guarda as 652 amostras de
treino e, para cada predição, calcula a distância até todas elas. Com
`weights='distance'`, ainda pondera o resultado. Nas 27 colunas do one-hot isso
são 652 × 27 operações por tabuleiro.

Na derivada cai para 4,19 ms, porque são 15 colunas e `weights='uniform'`.

> É o oposto dos demais algoritmos: o k-NN treina em 0,0254 s (quase nada, ele
> só memoriza) e paga tudo na predição. Para o front end, que classifica a cada
> jogada, essa é a métrica que importa.

### Overfitting?

Não há. A acurácia de teste é superior à de validação nas duas abordagens
(72,4% → 75,0% e 76,7% → 79,9%). O problema do k-NN aqui é o oposto:
**underfitting** — ele é simples demais para a estrutura do problema.

### Comparação com a execução exploratória

O notebook [02_knn_exploratorio.ipynb](../../../notebooks/02_knn_exploratorio.ipynb)
registrou um resultado bem pior, e a diferença é instrutiva:

| | Notebook | Script atual (`derivada`) |
|---|---|---|
| Pré-processamento | `StandardScaler` sobre os códigos 0/1/2 | features derivadas, padronizadas |
| Escolha do `k` | laço manual sobre a validação | `GridSearchCV` com `PredefinedSplit` |
| `k` escolhido | 19 | 5 |
| Acurácia no teste | 0,5488 | **0,7988** |

**Mais 25 pontos**, com o mesmo algoritmo e os mesmos dados. Duas causas:

1. **A representação.** Tratar `0`, `1` e `2` como escala numérica fazia a
   distância calcular coisas sem sentido ("X está a 2 unidades de vazio").
2. **O `k` menor.** Com `k=19` e apenas 9 empates no treino, era
   aritmeticamente impossível a classe Empate vencer uma votação — o modelo
   **nunca** a predizia (F1 = 0,00).

> Esse contraste é bom material para o relatório: mostra que *pré-processamento
> errado custa mais que algoritmo ruim*.

> Comparação com os demais algoritmos: [04-resultados.md](../04-resultados.md).

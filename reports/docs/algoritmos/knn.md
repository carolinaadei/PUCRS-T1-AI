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
| `bruta` | 72,4% | 78,0% | 0,776 | 0,780 | 0,771 | 0,0072 | 6,05 |
| `derivada` | 76,1% | **78,0%** | 0,797 | 0,780 | **0,782** | 0,0086 | 5,11 |

**Melhores hiperparâmetros**

| Abordagem | Configuração |
|---|---|
| `bruta` | `k=9` · euclidean · distance |
| `derivada` | `k=5` · euclidean · distance |

### Análise

**O k-NN é o pior dos cinco algoritmos** (78,0%, contra 89,0% do XGBoost), e é o
**único que não melhora com a abordagem derivada** — a acurácia fica idêntica
nas duas. O F1 sobe de leve (0,771 → 0,782), o que indica que ele distribui os
erros um pouco melhor entre as classes, mas não acerta mais.

Isso é coerente com a natureza do algoritmo. As features derivadas ajudam quem
consegue construir uma *fronteira* a partir delas. O k-NN não constrói fronteira
nenhuma: ele compara distâncias. Se dois tabuleiros de classes diferentes ficam
próximos no espaço de features — e ficam, porque uma única casa muda a classe —
nenhuma escolha de representação resolve.

**O `weights='distance'` venceu nas duas abordagens**, o que faz sentido: dar
mais peso ao vizinho mais próximo é a única defesa do k-NN contra vizinhanças
mistas.

**O `k` caiu de 9 para 5 na derivada.** Vizinhanças menores funcionam melhor
quando as features já separam as classes, porque não é preciso "votar" sobre uma
região grande para filtrar ruído.

### Comparação com a execução exploratória

O notebook [02_knn_exploratorio.ipynb](../../../notebooks/02_knn_exploratorio.ipynb)
registrou um resultado bem pior, e a diferença é instrutiva:

| | Notebook | Script atual (`bruta`) |
|---|---|---|
| Pré-processamento | `StandardScaler` sobre os códigos 0/1/2 | one-hot (27 colunas) |
| Escolha do `k` | laço manual sobre a validação | `GridSearchCV` com `PredefinedSplit` |
| `k` escolhido | 19 | 9 |
| Acurácia no teste | 0,5488 | **0,7800** |

**Mais 23 pontos**, com o mesmo algoritmo e os mesmos dados. Duas causas:

1. **O one-hot.** Tratar `0`, `1` e `2` como escala numérica fazia a distância
   euclidiana calcular coisas sem sentido ("X está a 2 unidades de vazio"). Com
   one-hot, a distância passa a contar *quantas casas diferem*, que é a noção
   correta de semelhança entre tabuleiros.
2. **O `k` menor.** Com `k=19` e apenas 9 empates no treino, era
   aritmeticamente impossível a classe Empate vencer uma votação — o modelo
   **nunca** a predizia (F1 = 0,00). Com `k=9`, ela passa a ser alcançável.

> Esse contraste é bom material para o relatório: mostra que *pré-processamento
> errado custa mais que algoritmo ruim*.

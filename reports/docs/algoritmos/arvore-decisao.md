# Árvore de Decisão

> Algoritmo exigido pelo enunciado.

**Código:** [models/arvore_decisao.py](../../../models/arvore_decisao.py)

```bash
python models/arvore_decisao.py                       # roda as duas abordagens
python models/arvore_decisao.py --abordagem derivada
```

---

## 1. Como funciona

A árvore aprende uma sequência de perguntas sobre as features. Em cada nó ela
escolhe a pergunta que melhor separa as classes — a que mais reduz a "impureza"
do conjunto — e divide os dados em dois ramos. Repete o processo em cada ramo
até que as folhas fiquem puras ou até bater um critério de parada.

Classificar um tabuleiro novo é percorrer a árvore da raiz até uma folha,
respondendo às perguntas.

O critério de impureza pode ser **Gini** ou **entropia**. Os dois medem o mesmo
fenômeno — quão misturadas estão as classes num nó — e costumam produzir árvores
parecidas; por isso ambos entram na grade.

**Vantagens**
- É o único modelo do trabalho que dá para **ler**: a árvore treinada é um
  conjunto de regras em português, o que facilita explicar o que foi aprendido.
- Não precisa de padronização — divide por limiares, não por distância.
- Lida naturalmente com features categóricas e com interações entre elas.

**Desvantagens**
- Sozinha, tende a decorar o treino se deixada crescer sem limite.
- Instável: mudar poucas amostras pode alterar bastante a árvore resultante.
- A fronteira de decisão é feita só de cortes paralelos aos eixos, o que é
  limitante quando a separação real é diagonal.

---

## 2. Pré-processamento

Rodado nas duas abordagens de [02-preprocessamento.md](../02-preprocessamento.md).

A árvore é o algoritmo **menos sensível** ao pré-processamento do trabalho: ela
não usa distância nem escala, então o `StandardScaler` da abordagem derivada é
indiferente para ela. O que muda de verdade, no caso dela, é *quais perguntas
ela pode fazer* — e é aí que a abordagem derivada tende a ajudar, como se
descreve abaixo.

---

## 3. Parâmetros testados

```python
PARAM_GRID = {
    "max_depth": [3, 5, 7, 10, None],
    "criterion": ["gini", "entropy"],
    "min_samples_leaf": [1, 3, 5],
}
```

| Parâmetro | Valores | Por quê |
|---|---|---|
| `max_depth` | 3, 5, 7, 10, `None` | é o principal controle de overfitting. Profundidades pequenas forçam generalização; `None` deixa crescer até o fim e serve como referência de quanto o modelo decoraria sem limite |
| `criterion` | `gini`, `entropy` | os dois critérios de impureza padrão; não há motivo teórico para preferir um aqui, então a validação decide |
| `min_samples_leaf` | 1, 3, 5 | exigir um mínimo de amostras por folha impede a árvore de criar folhas para casos isolados, que é a forma mais comum de decorar ruído |

São 5 × 2 × 3 = **30 combinações**.

> **Sobre a classe Empate.** Com `min_samples_leaf=5` e apenas 9 empates no
> treino, a árvore ficaria impedida de isolar essa classe em folha própria. Na
> prática a validação escolheu `min_samples_leaf=1` nas duas abordagens, então
> essa restrição não chegou a atuar.

---

## 4. Resultados

| Abordagem | Acur. val. | Acur. teste | Precision | Recall | F1 | Treino (s) | Predição (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bruta` | 57,1% | 54,9% | 0,550 | 0,549 | 0,549 | 0,0055 | 1,93 |
| `derivada` | 83,4% | **86,0%** | 0,845 | 0,860 | **0,849** | 0,0048 | 0,94 |

**Melhores hiperparâmetros**

| Abordagem | Configuração |
|---|---|
| `bruta` | gini · `max_depth=10` · `min_samples_leaf=1` |
| `derivada` | entropy · `max_depth=7` · `min_samples_leaf=1` |

### Análise — o resultado mais expressivo do trabalho

**Mais 31 pontos de acurácia, só trocando a representação da entrada.** Nenhum
outro algoritmo chega perto desse ganho, e a explicação é exata.

Na abordagem **bruta**, a árvore só consegue perguntar sobre casas individuais:
*"a casa do meio tem X?"*. Para reconhecer "três em linha" ela precisa aprender
a conjunção de três dessas perguntas — e precisa reaprender isso **oito vezes**,
uma para cada linha vencedora, porque cortes por casa não generalizam entre
linhas. Com 489 amostras de treino, não há dado suficiente para montar oito
sub-árvores dessas, e o resultado é 54,9%: pior que o k-NN, o pior do trabalho.

Na abordagem **derivada**, `linhas_2x` e `linhas_2o` respondem a mesma pergunta
em **um único corte**. A árvore salta para 86,0% e passa a empatar com o SVM.

### Sinais que confirmam a explicação

**A `max_depth` escolhida caiu de 10 para 7.** Na bruta a validação escolheu
quase a profundidade máxima da grade — a árvore precisou crescer muito para
compensar features pobres. Na derivada, menos profundidade basta.

**A bruta tem acurácia de teste (54,9%) *abaixo* da de validação (57,1%)**, e é a
única configuração das dez em que isso acontece de forma marcada. Combinado com
`max_depth=10` e `min_samples_leaf=1`, é o retrato de uma árvore que cresceu
para decorar e não generalizou.

**Na derivada, validação (83,4%) → teste (86,0%) sobe**, o padrão saudável.

### Custo

A árvore é **o algoritmo mais barato do trabalho**: 0,0048 s de treino e 0,94 ms
de predição na derivada — 17× mais rápida que o XGBoost para treinar, e a única
que prediz o conjunto de teste inteiro em menos de 1 ms.

Isso torna a árvore + derivada a escolha mais interessante sob critérios que não
sejam só acurácia: fica a 3 pontos do vencedor (86,0% contra 89,0%), custa uma
fração, e é **o único modelo do trabalho que dá para ler**.

### Visualizando a árvore

O notebook [01_construcao_dataset.ipynb](../../../notebooks/01_construcao_dataset.ipynb)
desenha os primeiros níveis com `plot_tree`, salvando em
`reports/figures/nb01_dt_arvore.png`. Com a abordagem derivada, os cortes do
topo devem ser sobre `linhas_2x` / `linhas_2o` — vale conferir, porque é a
confirmação visual de tudo que está nesta seção.

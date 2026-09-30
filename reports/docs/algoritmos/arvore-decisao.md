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

> **Atenção à classe Empate.** Com `min_samples_leaf=5` e apenas 9 empates no
> treino, a árvore fica impedida de isolar essa classe em folha própria. É um
> ponto a observar na matriz de confusão: se Empate desaparecer das predições,
> é provável que a configuração escolhida tenha `min_samples_leaf` alto.

---

## 4. Resultados

> **Pendente de execução.** Este algoritmo não tinha script próprio antes da
> reorganização — existia apenas como células exploratórias no notebook de
> construção do dataset, cujas saídas não foram preservadas. Rode
> `python models/arvore_decisao.py` e preencha a tabela abaixo com os valores de
> [reports/metrics/resultados.csv](../../metrics/).

| Abordagem | Acurácia val. | Acurácia teste | Precision | Recall | F1 | Treino (s) |
|---|---:|---:|---:|---:|---:|---:|
| `bruta` | | | | | | |
| `derivada` | | | | | | |

Melhores hiperparâmetros encontrados: _preencher_

Matriz de confusão: `reports/figures/arvore_de_decisao_<abordagem>_confusao.png`

### O que esperar, e o que verificar

**A abordagem derivada deve favorecer bastante a árvore.** Na abordagem bruta,
reconhecer "três em linha" exige que a árvore aprenda a conjunção de três
condições sobre casas específicas — e precisa reaprender isso para cada uma das
8 linhas vencedoras, porque não há como generalizar entre elas com cortes por
casa. Na abordagem derivada, `linhas_2x` e `linhas_2o` respondem isso em uma
única pergunta.

Vale verificar, ao rodar:

1. **A `max_depth` escolhida.** Se a validação escolher `None`, é sinal de que o
   modelo precisou de muita profundidade — o que, na abordagem bruta, seria
   consistente com a explicação acima.
2. **A classe Empate na matriz de confusão**, pelo motivo da seção 3.
3. **A distância entre acurácia de validação e de teste.** Uma queda grande
   indica que a árvore decorou.

### Visualizando a árvore

O notebook [01_construcao_dataset.ipynb](../../../notebooks/01_construcao_dataset.ipynb)
contém uma célula que desenha os primeiros níveis da árvore com `plot_tree`,
salvando em `reports/figures/nb01_dt_arvore.png`. É material útil para o
relatório: mostra, em regras legíveis, o que o modelo aprendeu.

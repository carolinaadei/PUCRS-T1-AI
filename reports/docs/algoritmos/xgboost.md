# XGBoost

> Algoritmo de **escolha livre** do grupo. O enunciado pede que os dois
> algoritmos livres venham acompanhados de uma explicação de como funcionam.

**Código:** [models/xgboost_clf.py](../../../models/xgboost_clf.py)

```bash
python models/xgboost_clf.py                      # roda as duas abordagens
python models/xgboost_clf.py --abordagem derivada
```

---

## 1. Como funciona

XGBoost (*Extreme Gradient Boosting*) é um método de **boosting**: em vez de
treinar um modelo forte, treina uma sequência de modelos fracos — árvores de
decisão rasas — onde **cada nova árvore tenta corrigir os erros das
anteriores**.

A mecânica é esta: a primeira árvore faz uma predição grosseira; calcula-se o
resíduo (o quanto ela errou em cada amostra); a segunda árvore é treinada para
prever esse resíduo; e assim por diante. A predição final é a soma das
contribuições de todas as árvores. O "gradient" do nome vem de o resíduo ser, na
verdade, o gradiente da função de perda.

A diferença do XGBoost para o *gradient boosting* clássico está na
**regularização**: ele penaliza explicitamente árvores complexas (via `gamma`,
`lambda` e limites de profundidade), o que o torna bem mais resistente a
overfitting do que uma sequência ingênua de árvores.

**Vantagens**
- Estado da arte em dados tabulares, que é o nosso caso.
- Regularização embutida e controlável.
- Captura relações não lineares e interações entre features.
- Não precisa de padronização — é feito de árvores.

**Desvantagens**
- Muitos hiperparâmetros interagindo entre si; a busca fica cara.
- Menos interpretável que uma árvore única.
- Dependência externa, fora do scikit-learn.
- Com poucos dados, o ganho sobre uma árvore bem regularizada pode não
  compensar o custo.

### Contraste com a Árvore de Decisão

Vale registrar, já que os dois algoritmos estão no trabalho: a
[árvore de decisão](arvore-decisao.md) é **uma** árvore que tenta resolver o
problema inteiro, e por isso tende a crescer e decorar. O XGBoost usa **muitas**
árvores propositalmente rasas (`max_depth` 3 a 7), cada uma resolvendo um pedaço
do erro restante. É a diferença entre um especialista e um comitê.

---

## 2. Pré-processamento

Rodado nas duas abordagens de [02-preprocessamento.md](../02-preprocessamento.md).

Como é feito de árvores, o XGBoost **não precisa** da padronização da abordagem
derivada — ela é inofensiva, mas não muda nada para ele. O que muda é o conjunto
de perguntas disponível, como na árvore de decisão.

---

## 3. Parâmetros testados

```python
MODELO = XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=RANDOM_STATE,
    n_jobs=1,
)

PARAM_GRID = {
    "learning_rate": [0.05, 0.1, 0.2],
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "gamma": [0, 0.1],
}
```

| Parâmetro | Valores | Por quê |
|---|---|---|
| `learning_rate` | 0,05 / 0,1 / 0,2 | quanto cada árvore contribui. Valores baixos exigem mais árvores mas generalizam melhor; é o principal par de troca com `n_estimators` |
| `n_estimators` | 100 / 200 | número de árvores. Mais árvores com `learning_rate` baixo costuma ser a combinação vencedora |
| `max_depth` | 3 / 5 / 7 | profundidade de cada árvore. No boosting, árvores **rasas** são o esperado: a força vem da soma, não de cada uma |
| `gamma` | 0 / 0,1 | ganho mínimo exigido para abrir uma divisão. `gamma > 0` poda divisões pouco úteis, regularizando |

Fixos, fora da grade:

| Parâmetro | Valor | Por quê |
|---|---|---|
| `objective` | `multi:softprob` | o problema tem **5 classes** |
| `eval_metric` | `mlogloss` | log-loss multiclasse, coerente com o objetivo |
| `n_jobs` | 1 | o paralelismo fica por conta do `GridSearchCV`, para não haver disputa por núcleos |

`num_class` é **omitido de propósito**: o wrapper scikit-learn do XGBoost o
infere dos rótulos, e defini-lo à mão conflita com essa inferência.

São 3 × 2 × 3 × 2 = **36 combinações**.

---

## 4. Resultados

**O XGBoost venceu a comparação do item 5**, nas duas abordagens.

| Abordagem | Acur. val. | Acur. teste | Precision | Recall | F1 | Treino (s) | Predição (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bruta` | 87,1% | 88,4% | 0,883 | 0,884 | **0,883** | 0,1160 | 4,84 |
| `derivada` | 85,9% | **89,0%** | 0,880 | 0,890 | 0,880 | 0,0799 | 2,62 |

**Melhores hiperparâmetros**

| Abordagem | Configuração |
|---|---|
| `bruta` | `learning_rate=0,2` · `max_depth=5` · `n_estimators=100` |
| `derivada` | `learning_rate=0,05` · `max_depth=3` · `n_estimators=100` |

### Análise

**89,0% no teste é o melhor resultado do trabalho**, e é o modelo servido pelo
[front end](../05-frontend.md). Mas a diferença para a abordagem bruta é de
**uma única amostra** (146 contra 145 acertos em 164), e o F1 é até ligeiramente
melhor na bruta — então não há vencedor claro entre as duas.

**A configuração derivada é a mais conservadora da grade:** `max_depth=3`, a
menor profundidade testada, e `learning_rate=0,05`, a menor taxa. É o
comportamento clássico de boosting bem ajustado — muitas árvores rasas somadas
com passos pequenos — e reforça que o resultado não vem de capacidade excessiva.

Na bruta, por contraste, o modelo precisou de árvores mais profundas
(`max_depth=5`) e passos maiores (`learning_rate=0,2`) para chegar quase ao
mesmo lugar. Mesma leitura dos outros algoritmos: sem as features prontas, é
preciso mais capacidade.

**`n_estimators=100` venceu nas duas**, e não 200. Com apenas 489 amostras de
treino, 100 árvores já saturam o que há para aprender.

### Overfitting?

Não há sinal em nenhuma das duas abordagens: a acurácia de teste é **superior**
à de validação nas duas (85,9% → 89,0% na derivada, 87,1% → 88,4% na bruta),
quando o padrão de um modelo que decorou seria o contrário.

Isso é consistente com a regularização embutida do XGBoost e com os valores
conservadores que a validação escolheu.

### Custo

| | `bruta` | `derivada` |
|---|---:|---:|
| Treino | 0,1160 s | **0,0799 s** (−31%) |
| Predição | 4,84 ms | **2,62 ms** (−46%) |

O XGBoost é o **segundo mais caro** para treinar, atrás do MLP — são 100 árvores
por configuração, contra um único modelo nos demais. Ainda assim, 0,08 s é
irrelevante na prática, e os 2,62 ms de predição não se notam no front end.

### O ponto cego da abordagem derivada

Com 89,0% de acurácia, a configuração vencedora **erra todos os 4 empates do
conjunto de teste**, sempre classificando-os como "X vence". A causa é uma
lacuna nas features derivadas — nenhuma das 15 colunas codifica "três em linha".
A análise completa, com as duas matrizes de confusão lado a lado, está em
[04-resultados.md §4.5](../04-resultados.md).

### A classe difícil

O F1 de "Possibilidade de Fim de Jogo" subiu de **0,76 para 0,90** — o melhor
resultado dos cinco algoritmos nessa classe.

### Sobre os números inválidos anteriores

A versão anterior do script usava `objective='binary:logistic'`, para **duas**
classes, num problema de **cinco**, e `cv=3` sobre o treino, ignorando o
conjunto de validação físico. Aquela execução registrou 0,7378 de acurácia.

Com o objetivo corrigido para `multi:softprob` e o protocolo unificado, o
resultado sobe para **0,8902** — **mais 15 pontos**, confirmando que o objetivo
errado estava de fato prejudicando o modelo. As figuras daquela execução
([xgboost_metricas.png](../../figures/xgboost_metricas.png),
[xgboost_matriz_confusao.png](../../figures/xgboost_matriz_confusao.png)) ficam
mantidas só para rastreabilidade e **não devem ser usadas no relatório**.

> Comparação com os demais algoritmos: [04-resultados.md](../04-resultados.md).

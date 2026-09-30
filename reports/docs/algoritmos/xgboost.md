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

> ⚠️ **Números inválidos — reexecutar.**
>
> A versão anterior deste script configurava `objective='binary:logistic'`, que é
> o objetivo para problemas de **duas** classes, enquanto o nosso tem **cinco**.
> As métricas registradas abaixo não descrevem o comportamento do modelo no
> problema real e **não devem ser usadas no relatório**.
>
> O script foi corrigido para `multi:softprob`. Rode
> `python models/xgboost_clf.py` e substitua esta seção pelos valores de
> [reports/metrics/resultados.csv](../../metrics/).

**Registro da execução inválida, para rastreabilidade**

| Parâmetro | Valor |
|---|---|
| `learning_rate` | 0,2 |
| `n_estimators` | 200 |
| `max_depth` | 3 |
| `gamma` | 0 |

| Métrica | Valor |
|---|---|
| Acurácia | 0,7378 |
| Precision (weighted) | 0,7333 |
| Recall (weighted) | 0,7378 |
| F1 (weighted) | 0,7340 |

Aquela execução também usava `cv=3` sobre o treino, ignorando o conjunto de
validação físico — um segundo desvio do item 4, independente do objetivo errado.

Figuras daquela execução:
[xgboost_metricas.png](../../figures/xgboost_metricas.png) ·
[xgboost_matriz_confusao.png](../../figures/xgboost_matriz_confusao.png)

### Tabela a preencher

| Abordagem | Acurácia val. | Acurácia teste | Precision | Recall | F1 | Treino (s) |
|---|---:|---:|---:|---:|---:|---:|
| `bruta` | | | | | | |
| `derivada` | | | | | | |

### O que verificar ao rodar

1. **Se o resultado sobe em relação a 0,7378.** É a evidência de que o objetivo
   errado estava de fato prejudicando o modelo.
2. **A `max_depth` escolhida.** Profundidade 3 seria o comportamento clássico de
   boosting; se a validação escolher 7, é sinal de que o problema exige árvores
   individuais mais expressivas.
3. **O custo de treino.** O XGBoost deve ser o mais caro do trabalho por larga
   margem — são até 200 árvores por configuração, contra um único modelo nos
   demais. Vale registrar isso na discussão de custo do item 3.
4. **A classe Empate.** Com 9 amostras de treino, é provável que as árvores
   simplesmente não criem divisões para ela.

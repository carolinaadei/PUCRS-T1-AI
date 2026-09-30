# 3. Metodologia — divisão dos dados e protocolo experimental

> Item 4 do enunciado — dividir fisicamente o conjunto em treino, validação e
> teste; usar os mesmos conjuntos em todos os experimentos; definir os
> parâmetros pela validação e reservar o teste para a avaliação final.

**Código:** [ttt/dataset.py](../../ttt/dataset.py) ·
[ttt/experiment.py](../../ttt/experiment.py) ·
**Dados:** [data/processed/](../../data/processed/)

---

## 3.1 Divisão física

Os três conjuntos são arquivos em disco, gerados uma única vez por
[notebooks/01_construcao_dataset.ipynb](../../notebooks/01_construcao_dataset.ipynb):

| Arquivo | Amostras | Papel |
|---|---:|---|
| [treino.csv](../../data/processed/treino.csv) | 489 | ajustar os pesos do modelo |
| [validacao.csv](../../data/processed/validacao.csv) | 163 | escolher os hiperparâmetros |
| [teste.csv](../../data/processed/teste.csv) | 164 | medir o desempenho final |

Serem arquivos, e não uma divisão recalculada dentro de cada script, é o que
garante a exigência de "mesmos conjuntos nos experimentos": os cinco algoritmos
leem exatamente as mesmas 489 linhas de treino.

A divisão é estratificada (`stratify=y`), então a proporção entre as classes se
mantém nos três conjuntos.

---

## 3.2 O protocolo

O mesmo para os cinco algoritmos, implementado em `treinar()` de
[ttt/experiment.py](../../ttt/experiment.py):

```
1. treino + validação entram no GridSearchCV, com PredefinedSplit
2. o melhor modelo é retreinado em treino + validação  (refit=True)
3. o conjunto de teste é tocado uma única vez, na avaliação final
```

### Passo 1 — `PredefinedSplit`, e não validação cruzada

Este é o ponto que mais importa, e o que mais costuma passar despercebido.

Se passássemos `cv=5` ao `GridSearchCV`, ele **ignoraria** o nosso arquivo de
validação e reparticionaria o treino aleatoriamente em 5 pedaços. O
`validacao.csv` viraria enfeite, e o enunciado pede explicitamente que ele seja
usado para definir os parâmetros.

`PredefinedSplit` resolve isso. A função `juntar_treino_validacao`, em
[ttt/dataset.py](../../ttt/dataset.py), empilha os dois conjuntos e devolve um
vetor `test_fold` que diz, linha a linha, qual é o papel de cada uma:

| valor | significado |
|---:|---|
| `-1` | linha de treino — nunca usada para pontuar |
| `0` | linha de validação — o único *fold* de avaliação |

O resultado é um "cross-validation" de uma única dobra, que é exatamente a
divisão física que definimos.

```python
X_tv, y_tv, test_fold = dataset.juntar_treino_validacao(dados)

busca = GridSearchCV(
    preprocessing.criar_pipeline(abordagem, modelo),
    grid,
    cv=PredefinedSplit(test_fold),   # <- a validação física, não folds aleatórios
    scoring="accuracy",
    refit=True,
)
```

### Passo 2 — Retreino em treino + validação

Com `refit=True`, depois de escolher os hiperparâmetros o `GridSearchCV`
retreina o modelo vencedor usando treino **e** validação. Faz sentido: uma vez
que a validação cumpriu o papel de escolher os parâmetros, descartar aquelas 163
amostras seria desperdiçar 20% dos dados rotulados.

### Passo 3 — O teste, uma vez só

O conjunto de teste aparece em exatamente uma linha do projeto:

```python
y_pred = busca.best_estimator_.predict(dados.X_teste)
```

Nada antes disso o consulta — nem o encoder, nem o scaler, nem a busca de
hiperparâmetros. É o que permite ler a acurácia de teste como estimativa de
desempenho em dados novos.

---

## 3.3 Métricas

As quatro que o enunciado pede, calculadas em `avaliar()` de
[ttt/evaluation.py](../../ttt/evaluation.py):

| Métrica | O que responde |
|---|---|
| Acurácia | que fração das predições está certa |
| Precision | entre os tabuleiros que o modelo chamou de classe X, quantos eram |
| Recall | entre os tabuleiros que eram classe X, quantos o modelo achou |
| F-measure | média harmônica entre precision e recall |

**As médias são `weighted`.** A razão está em [01-dataset.md](01-dataset.md):
Empate tem 4 amostras no teste contra 40 das demais classes. Uma média `macro`
daria a essas 4 amostras o mesmo peso das outras 160, e um único erro nelas
moveria a média geral mais do que dez erros em outra classe.

Além das médias, `avaliar()` imprime o `classification_report` **por classe**,
que é onde a análise de verdade acontece — ver os documentos de cada algoritmo
em [algoritmos/](algoritmos/).

---

## 3.4 O que cada execução produz

| Saída | Onde |
|---|---|
| Relatório por classe | console |
| Matriz de confusão | `reports/figures/<algoritmo>_<abordagem>_confusao.png` |
| Linha de métricas | [reports/metrics/resultados.csv](../metrics/) |
| Modelo treinado | `artifacts/<algoritmo>_<abordagem>.joblib` |

O CSV acumula uma linha por par (algoritmo, abordagem), substituindo a anterior
em cada reexecução para que não acumule lixo. É dele que
[models/comparar.py](../../models/comparar.py) monta a tabela e o gráfico
finais, e é dele que o front end descobre qual modelo carregar.

---

## 3.5 Reprodutibilidade

`random_state=42` está fixado em [ttt/config.py](../../ttt/config.py) e é usado
por todos os modelos que têm componente aleatório (árvore, MLP, SVM, XGBoost) e
pela divisão do dataset. Rodar duas vezes dá o mesmo resultado.

---

## 3.6 Cuidados contra overfitting

| Onde | O quê |
|---|---|
| Protocolo | teste isolado até a avaliação final |
| Dataset | deduplicação antes da divisão, para o mesmo tabuleiro não cair em treino e teste |
| Pipeline | pré-processamento reajustado por *fold*, sem vazar a validação |
| MLP | `early_stopping=True`, interrompendo o ajuste quando a perda estaciona |
| Árvore | `max_depth` e `min_samples_leaf` na grade, limitando o crescimento |
| SVM / XGBoost | `C` e `gamma` / `max_depth` e `gamma` na grade, controlando complexidade |

O sinal prático de que não houve overfitting severo é a proximidade entre a
acurácia de validação e a de teste. Quando a de teste desaba em relação à de
validação, o modelo decorou. Ver [04-resultados.md](04-resultados.md).

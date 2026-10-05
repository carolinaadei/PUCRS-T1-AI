# Documentação técnica

**T1 — Tic Tac Toe com Machine Learning**
PUCRS · Inteligência Artificial · Prof.ª Silvia Moraes
**Grupo:** Carolina Gonçalves · Vicente Goldani

Relatório do desenvolvimento, na ordem das etapas do enunciado. Cada documento
aponta para o código que o implementa.

| # | Documento | Item do enunciado |
|---|---|---|
| 1 | [Dataset](01-dataset.md) | Item 2 — obter, analisar e adequar o dataset do UCI |
| 2 | [Pré-processamento](02-preprocessamento.md) | Item 3 — as duas abordagens |
| 3 | [Metodologia](03-metodologia.md) | Item 4 — divisão dos dados e protocolo |
| 4 | [Resultados](04-resultados.md) | Item 5 — comparação e escolha do melhor |
| 5 | [Front end](05-frontend.md) | Item 6 — interação e acurácia ao vivo |

## Algoritmos

Um documento por algoritmo, com explicação do funcionamento, pré-processamento,
parâmetros testados e justificados, resultados e análise.

| Algoritmo | Documento | Código | Origem |
|---|---|---|---|
| k-NN | [knn.md](algoritmos/knn.md) | [models/knn.py](../../models/knn.py) | exigido |
| Árvore de Decisão | [arvore-decisao.md](algoritmos/arvore-decisao.md) | [models/arvore_decisao.py](../../models/arvore_decisao.py) | exigido |
| MLP | [mlp.md](algoritmos/mlp.md) | [models/mlp.py](../../models/mlp.py) | exigido |
| SVM | [svm.md](algoritmos/svm.md) | [models/svm.py](../../models/svm.py) | escolha livre |
| XGBoost | [xgboost.md](algoritmos/xgboost.md) | [models/xgboost_clf.py](../../models/xgboost_clf.py) | escolha livre |

---

## Resultado

**Melhor configuração: XGBoost com abordagem derivada — 89,0% de acurácia no
conjunto de teste** (F1 weighted 0,880). É o modelo servido pelo front end.

| Algoritmo | Melhor abordagem | Acur. teste | F1 |
|---|---|---:|---:|
| XGBoost | derivada | **89,0%** | 0,880 |
| SVM | bruta | 86,6% | 0,866 |
| Árvore de Decisão | derivada | 86,0% | 0,849 |
| MLP | derivada | 83,5% | 0,825 |
| k-NN | derivada | 78,0% | 0,782 |

**A abordagem derivada venceu as duas perguntas do item 3:** mais adequada
(84,5% contra 77,8% de acurácia média) e menos custosa (48% menos tempo de
treino). Tabela completa em [04-resultados.md](04-resultados.md).

### Os dois achados principais

**A representação importou mais que o algoritmo.** Trocar a entrada moveu a
árvore de decisão de 54,9% para 86,0% — 31 pontos. Trocar de algoritmo, mantida
a representação, move no máximo 11.

**A abordagem derivada não enxerga empates.** Nenhuma das 15 features codifica
"três em linha", então o melhor modelo erra todos os 4 empates do teste, e os 9
que apareceram em 100 partidas no front end. Ver
[04-resultados.md §4.5](04-resultados.md).

---

## Pendência

O `resultados.csv` da execução que gerou estes números **não está commitado** —
[../metrics/](../metrics/) só tem o CSV antigo do SVM, e [../figures/](../figures/)
não tem as matrizes de confusão novas. Ao rodar novamente, versione as duas
pastas para que os números tenham evidência rastreável:

```bash
python models/comparar.py
git add reports/metrics reports/figures
```

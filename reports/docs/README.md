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

## Estado das medições

Nem tudo foi executado sob o protocolo unificado. O que está medido e o que
falta:

| Algoritmo | Situação |
|---|---|
| MLP | medido na abordagem `bruta` (0,81 no teste) |
| k-NN | medido sob protocolo antigo (0,5488); reexecutar |
| SVM | medido sob protocolo antigo (~0,65); reexecutar |
| XGBoost | número inválido (objetivo binário em problema de 5 classes); reexecutar |
| Árvore de Decisão | nunca executado como script |

Para preencher tudo de uma vez:

```bash
pip install -r requirements.txt
python models/comparar.py
```

As tabelas marcadas como *pendente* nos documentos acima devem então ser
preenchidas com os valores de [../metrics/comparacao_final.csv](../metrics/).

---

## Figuras

Todas as figuras ficam em [../figures/](../figures/). As geradas pelos scripts
seguem o padrão `<algoritmo>_<abordagem>_confusao.png`; as geradas pelos
notebooks têm o prefixo `nb01_`.

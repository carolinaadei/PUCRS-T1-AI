# 4. Resultados e comparação

> Item 5 do enunciado — comparar os resultados, escolher o melhor algoritmo,
> mostrar a comparação em tabelas e gráficos, e justificar a escolha.

**Código:** [models/comparar.py](../../models/comparar.py) ·
**Dados:** [reports/metrics/](../metrics/) ·
**Figuras:** [reports/figures/](../figures/)

---

## 4.1 Como gerar

```bash
python models/comparar.py                  # roda os 5 algoritmos x 2 abordagens
python models/comparar.py --somente-tabela # só relê o CSV já existente
```

São 10 execuções (5 algoritmos × 2 abordagens). Cada uma grava uma linha em
[reports/metrics/resultados.csv](../metrics/) e uma matriz de confusão em
[reports/figures/](../figures/). Ao final, o script imprime a tabela
comparativa, salva `comparacao_final.csv` e o gráfico
`comparacao_algoritmos.png`.

---

## 4.2 Tabela comparativa

> **Pendente de execução completa.** Preencher com
> [reports/metrics/comparacao_final.csv](../metrics/) depois de rodar
> `python models/comparar.py`.

| Algoritmo | Abordagem | Acur. val. | Acur. teste | Precision | Recall | F1 | Treino (s) | Predição (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| k-NN | bruta | | | | | | | |
| k-NN | derivada | | | | | | | |
| Árvore de Decisão | bruta | | | | | | | |
| Árvore de Decisão | derivada | | | | | | | |
| MLP | bruta | | | | | | | |
| MLP | derivada | | | | | | | |
| SVM | bruta | | | | | | | |
| SVM | derivada | | | | | | | |
| XGBoost | bruta | | | | | | | |
| XGBoost | derivada | | | | | | | |

---

## 4.3 O que já está medido

Nem todos os algoritmos foram executados sob o protocolo unificado. O quadro
honesto, hoje, é este:

| Algoritmo | Acurácia no teste | Protocolo | Status |
|---|---:|---|---|
| [MLP](algoritmos/mlp.md) | **0,8100** | `PredefinedSplit` + one-hot | comparável |
| [k-NN](algoritmos/knn.md) | 0,5488 | laço manual + `StandardScaler` | **não** comparável |
| [SVM](algoritmos/svm.md) | ~0,65 | `cv=5` sobre o treino | **não** comparável |
| [XGBoost](algoritmos/xgboost.md) | 0,7378 | `cv=3` + objetivo binário | **inválido** |
| [Árvore de Decisão](algoritmos/arvore-decisao.md) | — | — | nunca executado como script |

Por isso a tabela de 4.2 está vazia em vez de preenchida com esses números: eles
foram obtidos com protocolos diferentes entre si, e compará-los diretamente
levaria a uma conclusão errada sobre qual algoritmo é melhor. Unificar o
protocolo foi exatamente o motivo da reorganização do projeto.

**O que dá para afirmar hoje:** o MLP com abordagem bruta atinge 81% de acurácia
no teste, com F1 weighted de 0,82, e é o único número medido sob o protocolo
correto.

---

## 4.4 Padrão que atravessa todos os algoritmos

Um resultado aparece em todo experimento medido até aqui, e é o achado mais
importante do trabalho:

**"Possibilidade de Fim de Jogo" é sistematicamente a classe mais difícil.**

| Algoritmo | F1 dessa classe | F1 da melhor classe |
|---|---:|---:|
| MLP | 0,65 | 0,94 (O vence) |
| k-NN | 0,39 | 0,67 (Tem jogo) |

A explicação é estrutural, e está em [01-dataset.md](01-dataset.md):

1. **É a única classe definida por uma contagem, não por uma posição.** As
   outras quatro se reconhecem olhando onde as marcas estão: três em linha,
   tabuleiro cheio, tabuleiro vazio-ish. Esta exige verificar *todas* as 8 linhas
   procurando o padrão "duas marcas iguais + uma casa vazia".
2. **É a classe com menor cobertura no dataset:** 200 amostras para 3.842
   estados que existem no jogo — 5,2%. As demais têm entre 30% e 100%.
3. **Ela faz fronteira com "Tem jogo".** Uma única jogada transforma um no
   outro, então tabuleiros muito parecidos pertencem a classes diferentes.

**Consequência prática:** é essa classe que a abordagem `derivada` deveria
resolver, porque `linhas_2x` e `linhas_2o` entregam a contagem pronta em vez de
exigir que o modelo a reconstrua. Verificar se o F1 dela sobe na abordagem
derivada é o teste mais informativo de todo o experimento.

### O segundo padrão: a classe Empate

Com 9 amostras de treino e 4 de teste, Empate é instável em todos os
algoritmos — e no k-NN ela simplesmente **nunca é predita** (F1 = 0,00), porque
com `k = 19` é aritmeticamente impossível que 9 amostras vençam uma votação.

Não é um defeito corrigível: 16 é o número total de empates que existem no jogo
da velha, e o dataset contém os 16. Ver [01-dataset.md](01-dataset.md), seção
1.4.

---

## 4.5 Comparação entre as abordagens (item 3)

> **Pendente de execução.** `python models/comparar.py` imprime este bloco e ele
> deve ser transcrito aqui:

```
Médias por abordagem de pré-processamento:
  <abordagem>  acurácia 0.xxxx | treino 0.xxxxs | predição xx.xxms
  <abordagem>  acurácia 0.xxxx | treino 0.xxxxs | predição xx.xxms
  → Mais adequada: '...'
  → Menos custosa: '...'
```

As duas perguntas do enunciado são independentes e podem ter respostas
diferentes:

- **Mais adequada** → maior acurácia média no teste.
- **Menos custosa** → menor tempo de treino do modelo escolhido. A abordagem
  `derivada` tem 15 colunas contra 27 da `bruta`, então a expectativa é que ela
  seja mais barata — mas ela paga o custo extra de calcular as features a cada
  predição, o que aparece em `tempo_predicao_ms`.

---

## 4.6 Como escolher o melhor algoritmo

O critério é a **acurácia no conjunto de teste**, que é o que
[models/comparar.py](../../models/comparar.py) usa para ordenar a tabela e
apontar o vencedor. Em caso de empate próximo, os critérios de desempate, em
ordem:

1. **F1 weighted**, que penaliza modelos que ignoram classes minoritárias — no
   nosso caso, exatamente o risco de um modelo abandonar "Empate".
2. **Distância entre acurácia de validação e de teste**, como sinal de
   overfitting: uma queda grande desqualifica.
3. **Custo**, pelo item 3.

O modelo vencedor é serializado em [artifacts/](../../artifacts/) e é o que o
front end carrega automaticamente — ver [05-frontend.md](05-frontend.md).

---

## 4.7 Figuras

| Figura | Conteúdo |
|---|---|
| `comparacao_algoritmos.png` | barras de acurácia por algoritmo e abordagem |
| `<algoritmo>_<abordagem>_confusao.png` | matriz de confusão de cada execução |
| [mlp_bruta_confusao.png](../figures/mlp_bruta_confusao.png) | matriz do MLP (medida) |
| [nb01_dt_arvore.png](../figures/) | primeiros níveis da árvore de decisão |

As matrizes de confusão são onde a análise por classe acontece: elas mostram
**com o quê** cada classe é confundida, e não apenas quanto se erra.

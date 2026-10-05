# 4. Resultados e comparação

> Item 5 do enunciado — comparar os resultados, escolher o melhor algoritmo,
> mostrar a comparação em tabelas e gráficos, e justificar a escolha.

**Código:** [models/comparar.py](../../models/comparar.py) ·
**Dados:** [reports/metrics/](../metrics/) ·
**Figuras:** [reports/figures/](../figures/)

> **Procedência dos números.** As tabelas abaixo são da execução completa de
> `python models/comparar.py` que gerou os slides da apresentação. O
> `resultados.csv` daquela execução ainda não foi commitado — ao rodar
> novamente, versione [reports/metrics/](../metrics/) para que os números
> tenham evidência rastreável no repositório.

---

## 4.1 Como gerar

```bash
python models/comparar.py                  # roda os 5 algoritmos x 2 abordagens
python models/comparar.py --somente-tabela # só relê o CSV já existente
```

São 10 execuções (5 algoritmos × 2 abordagens). Cada uma grava uma linha em
`reports/metrics/resultados.csv` e uma matriz de confusão em
[reports/figures/](../figures/).

---

## 4.2 Tabela comparativa

Ordenada por acurácia no conjunto de teste.

| Algoritmo | Abordagem | Acur. val. | Acur. teste | Precision | Recall | F1 | Treino (s) | Predição (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **XGBoost** | **derivada** | 85,9% | **89,0%** | 0,880 | 0,890 | 0,880 | 0,0708 | 4,29 |
| XGBoost | bruta | 87,1% | 88,4% | 0,883 | 0,884 | 0,883 | 0,1167 | 6,08 |
| SVM | bruta | 79,1% | 86,6% | 0,867 | 0,866 | 0,866 | 0,0439 | 9,46 |
| Árvore de Decisão | derivada | 83,4% | 86,0% | 0,845 | 0,860 | 0,849 | 0,0095 | 2,02 |
| SVM | derivada | 82,8% | 86,0% | 0,842 | 0,860 | 0,849 | 0,0142 | 3,99 |
| MLP | derivada | 82,2% | 83,5% | 0,815 | 0,835 | 0,825 | 0,0932 | 2,97 |
| MLP | bruta | 74,9% | 81,1% | 0,811 | 0,811 | 0,810 | 0,1989 | 4,54 |
| k-NN | bruta | 72,4% | 78,0% | 0,776 | 0,780 | 0,771 | 0,0072 | 6,05 |
| k-NN | derivada | 76,1% | 78,0% | 0,797 | 0,780 | 0,782 | 0,0086 | 5,11 |
| Árvore de Decisão | bruta | 57,1% | 54,9% | 0,550 | 0,549 | 0,549 | 0,0084 | 3,01 |

### Melhores hiperparâmetros

| Algoritmo | `bruta` | `derivada` |
|---|---|---|
| k-NN | k=9 · euclidean · distance | k=5 · euclidean · distance |
| Árvore de Decisão | gini · depth 10 · leaf 1 | entropy · depth 7 · leaf 1 |
| MLP | (50, 25) · relu · lr 0,01 | (50, 25) · tanh · lr 0,01 |
| SVM | rbf · C=10 · gamma scale | linear · C=0,1 |
| XGBoost | lr 0,2 · depth 5 · 100 árvores | lr 0,05 · depth 3 · 100 árvores |

---

## 4.3 Algoritmo escolhido: XGBoost com abordagem derivada

| | |
|---|---|
| Acurácia no teste | **89,0%** |
| F1 weighted | 0,880 |
| Validação → teste | 85,9% → 89,0% |
| Custo | treino 0,0708 s · predição 4,29 ms |
| Hiperparâmetros | `learning_rate=0.05`, `max_depth=3`, `n_estimators=100` |

**Justificativa:**

1. **Maior acurácia no teste** entre as dez configurações.
2. **Sem sinal de overfitting:** a acurácia de teste (89,0%) é *superior* à de
   validação (85,9%), o oposto do padrão de um modelo que decorou.
3. **`max_depth=3` e `learning_rate=0.05`** — a configuração vencedora é a mais
   *conservadora* da grade, não a mais expressiva. Isso é o comportamento
   clássico de boosting bem ajustado, e reforça que o resultado não vem de
   capacidade excessiva.
4. **Custo baixo:** 4,29 ms para classificar os 164 tabuleiros de teste, o que é
   irrelevante para o uso no front end.

É o modelo servido pelo [front end](05-frontend.md).

> **Ressalva honesta:** a diferença para o XGBoost `bruta` (88,4%) é de **uma
> única amostra** — 146 contra 145 acertos em 164. Não é uma diferença
> estatisticamente significativa, e a seção 4.5 mostra que as duas configurações
> erram de formas bem diferentes.

---

## 4.4 Comparação entre as abordagens (item 3)

| | `bruta` | `derivada` | |
|---|---:|---:|---|
| Acurácia média no teste | 77,8% | **84,5%** | +6,7 pontos |
| Tempo de treino médio | 0,075 s | **0,039 s** | −48% |
| Tempo de predição (164 tabuleiros) | 5,83 ms | **3,68 ms** | −37% |

**A abordagem derivada venceu nas duas perguntas do enunciado: é mais adequada
e menos custosa.** É um resultado limpo, porque normalmente há um compromisso
entre as duas — aqui não há. A explicação é direta: 15 colunas contra 27, e as
15 já carregam a regra do jogo, então o modelo precisa de menos capacidade para
chegar mais longe.

### Onde o ganho aconteceu: a classe difícil

O F1 de "Possibilidade de Fim de Jogo", por algoritmo:

| Algoritmo | `bruta` | `derivada` | Ganho |
|---|---:|---:|---:|
| k-NN | 0,59 | 0,76 | +0,17 |
| Árvore de Decisão | 0,47 | 0,87 | **+0,40** |
| MLP | 0,65 | 0,84 | +0,19 |
| SVM | 0,78 | 0,89 | +0,11 |
| XGBoost | 0,76 | 0,90 | +0,14 |

**A hipótese levantada em [02-preprocessamento.md](02-preprocessamento.md) se
confirmou.** Aquela classe é definida pela condição "existe linha com duas
marcas do mesmo jogador e a terceira casa vazia". Na abordagem bruta o modelo
tem que reconstruir essa regra a partir das 27 colunas; na derivada ela chega
pronta em `linhas_2x` e `linhas_2o`. Todos os cinco algoritmos melhoraram, sem
exceção.

### O caso extremo: a Árvore de Decisão

| | `bruta` | `derivada` |
|---|---:|---:|
| Acurácia no teste | 54,9% | **86,0%** |

**Mais 31 pontos, só trocando a representação da entrada.** É o resultado mais
expressivo do trabalho, e tem explicação exata.

Na abordagem bruta, a árvore só pode perguntar sobre casas individuais
(`a casa do meio tem X?`). Para reconhecer "três em linha" ela precisa aprender
a conjunção de três condições — e precisa reaprender isso **oito vezes**, uma
para cada linha vencedora, porque não há como generalizar entre elas com cortes
por casa. Com `max_depth` limitado, não cabe.

Na abordagem derivada, `linhas_2x` e `linhas_2o` respondem isso em **uma
pergunta**. A árvore deixa de ser o pior algoritmo do trabalho e passa a empatar
com o SVM.

> **Conclusão que vale para o relatório:** neste problema, **a representação
> importou mais que a escolha do algoritmo**. Trocar a entrada moveu a árvore em
> 31 pontos; trocar de algoritmo, mantida a representação, move no máximo 11
> pontos (de 78,0% do k-NN a 89,0% do XGBoost, na derivada).

---

## 4.5 O preço da abordagem derivada: ela não enxerga empates

As duas matrizes de confusão do XGBoost, que estão separadas por uma única
amostra de acurácia, revelam comportamentos bem diferentes.

**XGBoost · derivada — 89,0%**

| real \ predito | Empate | O vence | Possib. | Tem jogo | X vence |
|---|---:|---:|---:|---:|---:|
| **Empate** | 0 | 0 | 0 | 0 | **4** |
| **O vence** | 0 | 40 | 0 | 0 | 0 |
| **Possibilidade** | 0 | 2 | 33 | 0 | 5 |
| **Tem jogo** | 0 | 4 | 0 | 35 | 1 |
| **X vence** | 0 | 0 | 0 | 2 | 38 |

**XGBoost · bruta — 88,4%**

| real \ predito | Empate | O vence | Possib. | Tem jogo | X vence |
|---|---:|---:|---:|---:|---:|
| **Empate** | **4** | 0 | 0 | 0 | 0 |
| **O vence** | 0 | 40 | 0 | 0 | 0 |
| **Possibilidade** | 0 | 0 | 29 | **10** | 1 |
| **Tem jogo** | 0 | 0 | **7** | 32 | 1 |
| **X vence** | 0 | 0 | 0 | 0 | 40 |

### A leitura

**A derivada erra todos os 4 empates, e sempre para "X vence".**

O motivo é uma lacuna no conjunto de features: **nenhuma das 15 colunas diz se
existe três em linha.** Elas contam marcas, contam ameaças e dizem de quem é a
vez — mas não codificam vitória. Para o modelo, um tabuleiro cheio sem vencedor
(empate) e um tabuleiro cheio com X vencendo têm features quase idênticas:
`qtd_x=5`, `qtd_o=4`, `casas_vazias=0`, `linhas_2x=0`, `linhas_2o=0`, todas as
casas ocupadas. Como "X vence" tem 200 amostras e "Empate" tem 16, o modelo
aposta sempre na classe majoritária — e acerta 98% das vezes.

**A bruta acerta os 4 empates**, porque o one-hot das 9 casas preserva a
informação de *quais* marcas estão onde, que é o necessário para ver três em
linha. Em compensação, ela confunde Possibilidade ↔ Tem jogo **17 vezes**,
exatamente a fronteira que a derivada resolve.

### Por que isso é a melhor análise do trabalho

As duas abordagens não são "uma melhor e outra pior": elas **acertam coisas
diferentes** e empatam na média por coincidência. Cada representação torna uma
informação explícita e outra invisível:

| | `bruta` | `derivada` |
|---|---|---|
| Enxerga três em linha | sim | **não** |
| Enxerga ameaça (2 + vazia) | precisa deduzir | sim, pronta |
| Erra | Possib. ↔ Tem jogo (17×) | todos os empates (4×) |

**O próximo passo óbvio** é acrescentar às features derivadas uma coluna
`tres_em_linha` (ou `vencedor`), o que deveria recuperar os empates sem perder o
ganho na classe difícil. A função `vencedor` já existe em
[ttt/game_rules.py](../../ttt/game_rules.py) — seria uma linha em
`_features_do_tabuleiro`, em [ttt/preprocessing.py](../../ttt/preprocessing.py).

---

## 4.6 Outras observações

**O k-NN é o pior dos cinco (78,0%), e é o único que não melhora com a
derivada** — a acurácia fica idêntica nas duas abordagens, embora o F1 suba
levemente (0,771 → 0,782). Ver [algoritmos/knn.md](algoritmos/knn.md).

**O SVM foi o único a preferir a abordagem bruta** (86,6% contra 86,0%), e com
uma inversão interessante de kernel: `rbf` na bruta, `linear` com `C=0,1` na
derivada. Faz sentido — com as features já separando bem as classes, um
hiperplano simples basta, e `C` baixo indica que a margem larga foi suficiente.

**O MLP é o mais caro para treinar** (0,1989 s na bruta), quase 20× o custo da
árvore, para um resultado 5 pontos pior que o XGBoost.

**A árvore é a mais barata de todas** (0,0095 s de treino, 2,02 ms de predição) e,
com a abordagem derivada, chega a 86,0% — dentro de 3 pontos do vencedor. Se o
critério incluísse custo e interpretabilidade, ela seria a escolha.

---

## 4.7 Figuras

| Figura | Conteúdo |
|---|---|
| `comparacao_algoritmos.png` | barras de acurácia por algoritmo e abordagem |
| `<algoritmo>_<abordagem>_confusao.png` | matriz de confusão de cada execução |
| [nb01_dt_arvore.png](../figures/) | primeiros níveis da árvore de decisão |

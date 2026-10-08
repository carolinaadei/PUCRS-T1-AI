# 2. Pré-processamento — as duas abordagens

> Item 3 do enunciado — testar duas abordagens de pré-processamento e verificar
> qual é **mais adequada para o problema e menos custosa**.

**Código:** [ttt/preprocessing.py](../../ttt/preprocessing.py) ·
**Features derivadas:** função `_features_do_tabuleiro` ·
**Regras usadas:** [ttt/game_rules.py](../../ttt/game_rules.py)

---

## 2.1 A ideia

As duas abordagens diferem em **uma coisa só**: o que o modelo recebe como
entrada. O algoritmo, os conjuntos de dados, a grade de hiperparâmetros e o
protocolo de avaliação são idênticos. Sem isso a comparação não diria nada sobre
a representação — diria sobre o resto.

Por isso as duas vivem no mesmo módulo, atrás da mesma função:

```python
criar_pipeline("bruta", modelo)      # -> Pipeline(onehot, modelo)
criar_pipeline("derivada", modelo)   # -> Pipeline(features, escala, modelo)
```

Em ambos os casos o modelo entra **dentro** do `Pipeline`. Isso não é detalhe de
organização: faz o pré-processamento ser reajustado a cada *fold* do
`GridSearchCV`, de modo que ele nunca veja os dados de validação antes da hora.
Se o encoder fosse ajustado uma vez sobre tudo, haveria vazamento.

---

## 2.2 Abordagem `bruta` — o tabuleiro atual

A entrada é o tabuleiro como ele é: 9 casas com os valores `0` (vazio), `1` (O)
e `2` (X).

Sobre esses 9 valores aplicamos **one-hot**, e cada casa vira 3 colunas
(vazio / O / X), totalizando **27 colunas**.

**Por que one-hot e não os códigos direto?** Porque `0`, `1` e `2` são
*categorias*, não uma escala. Entregues crus, algoritmos que calculam distância
ou combinação linear leriam "X é o dobro de O" e "vazio é metade de O", o que não
tem significado no jogo. O one-hot elimina essa ordem inventada.

```python
etapas = [("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))]
```

`handle_unknown="ignore"` evita quebra se alguma partição não contiver os três
valores em alguma casa.

---

## 2.3 Abordagem `derivada` — features do tabuleiro

A entrada passa a ser um resumo do tabuleiro, com as features que o enunciado
lista. São **15 colunas**:

| Feature | Descrição |
|---|---|
| `qtd_x` | quantidade de X |
| `qtd_o` | quantidade de O |
| `casas_vazias` | número de casas vazias |
| `linhas_2x` | linhas com 2 X e a terceira casa vazia |
| `linhas_2o` | linhas com 2 O e a terceira casa vazia |
| `jogador_da_vez` | de quem é a vez (1 = O, 2 = X) |
| `ocupada_0` … `ocupada_8` | máscara binária das posições ocupadas |

Duas decisões de interpretação, ambas registradas em código:

**"Posições ocupadas" virou uma máscara binária de 9 casas, não uma contagem.**
A contagem já é dada por `casas_vazias` (são complementares), então uma segunda
contagem seria informação redundante. A máscara, sim, acrescenta algo novo: a
*geometria* de onde as marcas estão, sem dizer de quem são.

**`jogador_da_vez` é derivado, não informado.** X abre a partida, então joga
sempre que os dois jogadores têm o mesmo número de marcas. Está em
`jogador_da_vez`, em [ttt/game_rules.py](../../ttt/game_rules.py).

Depois da extração, as features passam por `StandardScaler`:

```python
etapas = [
    ("features", FunctionTransformer(extrair_features)),
    ("escala", StandardScaler()),
]
```

**Por que padronizar aqui e não na abordagem bruta?** Porque aqui as escalas
convivem: `casas_vazias` vai de 0 a 9, `linhas_2x` de 0 a ~3, e as 9 flags
`ocupada_*` só valem 0 ou 1. Para k-NN e SVM, que decidem por distância, a
feature de maior amplitude dominaria o cálculo. Na abordagem bruta o one-hot já
produz só zeros e uns, então não há escala a corrigir.

---

## 2.4 O que cada abordagem aposta

| | `bruta` | `derivada` |
|---|---|---|
| Colunas | 27 | 15 |
| Conhecimento do jogo embutido | nenhum | as regras, via `linhas_com_duas` e `jogador_da_vez` |
| O que o modelo precisa aprender | tudo, inclusive o que é "três em linha" | só a fronteira entre as classes |

A abordagem `bruta` entrega o tabuleiro e deixa o modelo descobrir os padrões.
A `derivada` já entrega o padrão pronto.

Isso importa especialmente para a classe **Possibilidade de Fim de Jogo**, que é
a única definida por uma condição que não se lê diretamente das casas, mas de
uma contagem sobre elas: *existe alguma linha com duas marcas do mesmo jogador e
a terceira casa vazia?* Na abordagem bruta o modelo tem que reconstruir essa
regra a partir das 27 colunas. Na derivada ela chega pronta, em `linhas_2x` e
`linhas_2o`.

---

## 2.5 Como comparar

Qualquer script de algoritmo roda as duas abordagens por padrão:

```bash
python models/knn.py                     # bruta e derivada, em sequência
python models/knn.py --abordagem bruta   # só uma delas
```

Ao final, ele imprime qual foi mais precisa e qual foi mais barata. A visão
consolidada dos cinco algoritmos sai de:

```bash
python models/comparar.py
```

que grava [reports/metrics/resultados.csv](../metrics/) e responde às duas
perguntas do item 3 com médias por abordagem:

```
Médias por abordagem de pré-processamento:
  <abordagem>  acurácia 0.xxxx | treino 0.xxxxs | predição xx.xxms
  → Mais adequada: '...'
  → Menos custosa: '...'
```

### Como o custo é medido

"Menos custosa" é medido em [ttt/experiment.py](../../ttt/experiment.py), com
duas grandezas:

- **`tempo_treino_s`** — vem de `GridSearchCV.refit_time_`, que é o tempo de
  treinar **apenas o modelo escolhido** sobre treino + validação. Não inclui a
  busca de hiperparâmetros, que depende do tamanho da grade e não da
  representação, e portanto poluiria a comparação.
- **`tempo_predicao_ms`** — tempo de classificar o conjunto de teste inteiro
  (164 tabuleiros). É o custo que aparece no front end, onde a predição acontece
  a cada jogada.

### Resposta medida

| | `bruta` | `derivada` | |
|---|---:|---:|---|
| Acurácia média no teste | 77,2% | **84,9%** | +7,7 pontos |
| Tempo de treino médio | 0,0581 s | **0,0303 s** | −48% |
| Tempo de predição (164 tabuleiros) | 11,58 ms | **2,47 ms** | −79% |

**A abordagem derivada é, ao mesmo tempo, a mais adequada e a menos custosa** —
as duas perguntas do item 3 têm a mesma resposta.

Mas ela tem um ponto cego: **nenhuma das 15 features codifica "três em linha"**,
e por isso o melhor modelo erra todos os empates. A análise completa, com as
matrizes de confusão, está em [04-resultados.md §4.5](04-resultados.md).

> **Resultados completos:** [04-resultados.md](04-resultados.md).

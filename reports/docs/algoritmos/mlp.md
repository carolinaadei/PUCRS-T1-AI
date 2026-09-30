# Multi-Layer Perceptron (MLP)

> Algoritmo exigido pelo enunciado.

**Código:** [models/mlp.py](../../../models/mlp.py)

```bash
python models/mlp.py                      # roda as duas abordagens
python models/mlp.py --abordagem bruta
```

---

## 1. Como funciona

O MLP é uma rede neural *feedforward*: neurônios organizados em camadas, onde
cada neurônio de uma camada se conecta a todos os da camada seguinte. Cada
conexão tem um peso.

A entrada atravessa a rede camada por camada — cada neurônio soma suas entradas
ponderadas e aplica uma função de ativação não linear — até a camada de saída,
que produz uma pontuação por classe. O treino usa *backpropagation*: compara a
saída com o rótulo correto, calcula o erro e propaga ajustes nos pesos de trás
para frente.

A não linearidade das ativações é o que diferencia o MLP de um modelo linear:
sem ela, empilhar camadas seria equivalente a uma única transformação linear.

**Vantagens**
- Aproxima fronteiras de decisão de forma muito flexível.
- Captura interações entre features sem que elas sejam declaradas.

**Desvantagens**
- Caixa-preta: não dá para ler o que foi aprendido, ao contrário da árvore.
- Sensível à escala das entradas e à inicialização dos pesos.
- Muitos hiperparâmetros, e o treino é mais caro que o dos demais.
- Precisa de bastante dado; com 489 amostras de treino, decora com facilidade.

---

## 2. Topologia e pré-processamento

> O enunciado pede explicitamente a topologia do MLP.

A topologia não é fixada à mão: entra na grade e é escolhida pelo conjunto de
validação. As quatro candidatas são:

| `hidden_layer_sizes` | Topologia |
|---|---|
| `(10,)` | 1 camada oculta de 10 neurônios |
| `(50,)` | 1 camada oculta de 50 neurônios |
| `(10, 10)` | 2 camadas ocultas de 10 neurônios cada |
| `(50, 25)` | 2 camadas ocultas, 50 e 25 neurônios |

As camadas de entrada e saída são determinadas pelo problema:

| | Abordagem `bruta` | Abordagem `derivada` |
|---|---|---|
| Entrada | 27 neurônios (one-hot das 9 casas) | 15 neurônios (features extraídas) |
| Saída | 5 neurônios (uma por classe) | 5 neurônios |

Ou seja, a topologia vencedora na execução registrada abaixo foi
**27 → 50 → 25 → 5**.

O MLP é sensível à escala, então as duas abordagens de
[02-preprocessamento.md](../02-preprocessamento.md) já entregam entradas
adequadas: one-hot (valores 0/1) na bruta, padronização na derivada.

### Nota sobre o alvo

O alvo permanece **numérico** durante todo o treinamento. O `early_stopping`
interno do `MLPClassifier` é incompatível com rótulos em texto — ele tentaria
aplicar `np.isnan` sobre strings e quebraria. A tradução para nomes legíveis
acontece só na avaliação, via `CLASS_MAP` em
[ttt/config.py](../../../ttt/config.py).

---

## 3. Parâmetros testados

```python
MODELO = MLPClassifier(
    max_iter=2000,
    early_stopping=True,
    n_iter_no_change=20,
    random_state=RANDOM_STATE,
)

PARAM_GRID = {
    "hidden_layer_sizes": [(10,), (50,), (10, 10), (50, 25)],
    "activation": ["relu", "tanh"],
    "learning_rate_init": [0.001, 0.01],
}
```

| Parâmetro | Valores | Por quê |
|---|---|---|
| `hidden_layer_sizes` | 4 topologias | cobre redes rasas e profundas, estreitas e largas, para medir se o problema exige capacidade |
| `activation` | `relu`, `tanh` | ReLU é o padrão moderno e treina mais rápido; tanh é saturante e às vezes se sai melhor em redes pequenas |
| `learning_rate_init` | 0,001 / 0,01 | taxa baixa converge mais estável; taxa alta converge mais rápido mas pode oscilar |

Fixos, fora da grade:

| Parâmetro | Valor | Por quê |
|---|---|---|
| `max_iter` | 2000 | teto alto o bastante para o `early_stopping` decidir a parada, e não o limite de iterações |
| `early_stopping` | `True` | **principal defesa contra overfitting.** Reserva parte do treino para monitorar a perda e interrompe quando ela para de melhorar |
| `n_iter_no_change` | 20 | quantas épocas sem melhora toleradas antes de parar |

São 4 × 2 × 2 = **16 combinações**.

---

## 4. Resultados

Execução registrada na abordagem `bruta` (one-hot, `PredefinedSplit`):

**Melhor configuração**

```python
{'activation': 'relu', 'hidden_layer_sizes': (50, 25), 'learning_rate_init': 0.01}
```

| | |
|---|---|
| Acurácia na validação | 0,7485 |
| **Acurácia no teste** | **0,8100** |
| F1 macro | 0,82 |

**Por classe, no conjunto de teste**

| Classe | Precision | Recall | F1 | Suporte |
|---|---:|---:|---:|---:|
| Empate | 1,00 | 0,75 | 0,86 | 4 |
| O vence | 0,91 | 0,97 | 0,94 | 40 |
| Possibilidade de Fim de Jogo | 0,65 | 0,65 | 0,65 | 40 |
| Tem jogo | 0,79 | 0,78 | 0,78 | 40 |
| X vence | 0,87 | 0,85 | 0,86 | 40 |

Matriz de confusão: [mlp_bruta_confusao.png](../../figures/mlp_bruta_confusao.png)

### Análise

**O desempenho foi bom.** 81% num problema de 5 classes, contra 20% do acaso, e
o melhor resultado entre os algoritmos com números medidos até agora.

**"O vence" é a classe mais fácil (F1 = 0,94).** Três em linha de O é um padrão
geométrico fixo, que aparece em 8 configurações possíveis, e a rede aprende bem
com 120 exemplos de treino.

**"Possibilidade de Fim de Jogo" é a mais difícil (F1 = 0,65),** com precision e
recall igualmente baixos — o que indica confusão **simétrica**: o modelo tanto
rotula outras classes como esta quanto deixa passar exemplos que são dela. A
explicação está em [01-dataset.md](../01-dataset.md): é a única classe definida
por uma contagem sobre as casas, não pela posição delas, e é a classe com menor
cobertura no dataset (5,2% dos estados que existem no jogo). Na abordagem bruta
a rede precisa reconstruir essa regra a partir das 27 colunas de entrada.

**"Empate" com precision 1,00 e recall 0,75 merece cautela.** São 4 amostras no
teste: o modelo acertou 3 e errou 1, e nenhum falso positivo. O F1 de 0,86 é
matematicamente correto, mas estatisticamente frágil — não é base para concluir
que o modelo domina essa classe.

### Overfitting?

Não há sinal de overfitting severo:

- A acurácia de validação (0,7485) e a de teste (0,81) são compatíveis, e a de
  teste foi até **superior** — o oposto do padrão de um modelo que decorou.
- O `early_stopping` atuou durante toda a busca, impedindo que a rede continuasse
  ajustando depois que a perda estacionou.

Com 816 amostras no total, porém, não dá para descartar completamente. Técnicas
como *dropout* não estão disponíveis no `MLPClassifier` do scikit-learn e
exigiriam PyTorch ou Keras.

### Contra as expectativas

| Expectativa | Resultado |
|---|---|
| Acurácia acima de 70% | superada: 81% |
| Rede de 2 camadas melhor que 1 | confirmado: `(50, 25)` > `(50,)` > `(10, 10)` > `(10,)` |
| ReLU e tanh equivalentes | ReLU foi superior |
| Taxa 0,01 mais instável que 0,001 | não confirmado: 0,01 foi melhor, provavelmente porque o `early_stopping` absorve a oscilação |

### Próximo passo

A hipótese sobre "Possibilidade de Fim de Jogo" é testável: na abordagem
`derivada`, `linhas_2x` e `linhas_2o` entregam a condição pronta. Rodar
`python models/mlp.py` mede as duas abordagens e mostra se o F1 dessa classe
sobe.

> Os números acima são da abordagem `bruta`. Os da abordagem `derivada` e a
> comparação com os demais algoritmos saem de `python models/comparar.py` —
> ver [04-resultados.md](../04-resultados.md).

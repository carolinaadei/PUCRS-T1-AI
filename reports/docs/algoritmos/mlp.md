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

A topologia `(50, 25)` venceu nas duas abordagens, o que dá
**27 → 50 → 25 → 5** na bruta e **15 → 50 → 25 → 5** na derivada.

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

| Abordagem | Acur. val. | Acur. teste | Precision | Recall | F1 | Treino (s) | Predição (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bruta` | 74,8% | 81,1% | 0,811 | 0,811 | 0,810 | 0,1213 | 2,17 |
| `derivada` | 82,2% | **83,5%** | 0,815 | 0,835 | **0,825** | 0,0518 | 2,33 |

**Melhores hiperparâmetros**

| Abordagem | Topologia | Ativação | Taxa inicial |
|---|---|---|---|
| `bruta` | (50, 25) | relu | 0,01 |
| `derivada` | (50, 25) | tanh | 0,01 |

### Topologia escolhida (derivada)

```
15 features  →  50 neurônios (tanh)  →  25 neurônios (tanh)  →  5 classes (softmax)
```

| | |
|---|---|
| Otimizador | Adam, `learning_rate_init = 0,01` |
| Parada | `early_stopping`, 20 épocas sem melhora |
| Limite | `max_iter = 2000` |

Na abordagem bruta a topologia é a mesma, com 27 neurônios na entrada em vez de
15.

### Análise

**A topologia `(50, 25)` venceu nas duas abordagens**, confirmando a expectativa
de que o problema exige mais capacidade que uma camada estreita. A taxa de
aprendizado 0,01 também venceu nas duas — mais alta que o usual, o que o
`early_stopping` absorve sem instabilidade.

**A ativação mudou: relu na bruta, tanh na derivada.** É coerente com a natureza
das entradas. Na bruta, o one-hot entrega zeros e uns, e a ReLU lida bem com
entradas esparsas. Na derivada, o `StandardScaler` entrega valores centrados em
zero e com sinal, que é exatamente o domínio onde a tanh (também centrada em
zero, saturando em ±1) se sai melhor.

**O MLP é o algoritmo mais caro para treinar: 0,1213 s na bruta**, 22× o custo
da árvore, para um resultado 5 pontos pior que o XGBoost. A abordagem derivada
corta esse custo pela metade (0,0518 s), porque a rede tem 15 entradas em vez de
27 — menos pesos na primeira camada.

**Ganho moderado com a derivada (+2,4 pontos),** menor que o da árvore (+31) ou
do MLP na classe difícil. Faz sentido: a rede já conseguia aproximar a regra de
"três em linha" a partir do one-hot, então entregá-la pronta ajuda, mas não
transforma o resultado.

### Overfitting?

Não há sinal nas duas abordagens. A acurácia de teste é **superior** à de
validação em ambas (74,8% → 81,1% e 82,2% → 83,5%), o oposto do padrão de um
modelo que decorou. O `early_stopping` atuou durante toda a busca.

### Contra as expectativas

| Expectativa | Resultado |
|---|---|
| Acurácia acima de 70% | superada nas duas abordagens |
| Rede de 2 camadas melhor que 1 | confirmado: `(50, 25)` venceu nas duas |
| ReLU e tanh equivalentes | depende do pré-processamento: relu na bruta, tanh na derivada |
| Taxa 0,01 mais instável que 0,001 | não confirmado: 0,01 venceu nas duas |

### A classe difícil

O F1 de "Possibilidade de Fim de Jogo" subiu de **0,65 para 0,84** com a
abordagem derivada — o segundo maior ganho do trabalho, atrás apenas da árvore.
A hipótese levantada aqui antes da execução se confirmou: entregar `linhas_2x` e
`linhas_2o` prontas resolve boa parte dessa classe.

> Comparação com os demais algoritmos: [04-resultados.md](../04-resultados.md).

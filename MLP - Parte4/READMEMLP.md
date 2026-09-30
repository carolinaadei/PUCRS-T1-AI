Aqui vai uma versão reescrita, com linguagem mais natural e menos “engessada”, mantendo o conteúdo técnico:

---

# Documentação — MLP para Classificação de Jogo da Velha

**Disciplina:** Inteligência Artificial
**Algoritmo:** Multi-Layer Perceptron (MLP)
**Problema:** Classificação de estados de um tabuleiro 3×3 de Jogo da Velha
**Arquivo principal:** `Parte4/algoritmoMLP.py`

---

## 1. Visão geral do algoritmo

### Como o modelo funciona

O **MLP (Multi-Layer Perceptron)** é uma rede neural totalmente conectada, usada em problemas de aprendizado supervisionado. A ideia básica é receber um conjunto de entradas, transformá-las ao longo de algumas camadas internas e, no final, produzir uma saída com a previsão.

A estrutura utilizada pode ser representada assim:

```
Entrada (27 neurônios) → Camadas ocultas → Saída (5 neurônios)
```

O funcionamento acontece em dois momentos principais:

* **Forward pass:** os dados entram na rede, passam por cada camada, são combinados com pesos e passam por funções de ativação (como ReLU ou tanh).
* **Backpropagation:** o erro entre a saída prevista e a real é calculado e usado para ajustar os pesos da rede.

O treinamento foi feito com o otimizador **Adam**, que ajusta automaticamente a taxa de aprendizado de cada parâmetro, facilitando a convergência.

### Onde esse modelo se encaixa

O MLP é bem versátil e pode ser usado para:

* Classificação (binária ou multiclasse)
* Regressão
* Reconhecimento de padrões

Neste trabalho, ele foi usado para classificar o estado de um tabuleiro com base nas suas casas, sem regras explícitas do jogo.

### Pontos positivos e limitações

**Vantagens:**

* Consegue capturar relações não-lineares
* Estrutura flexível (camadas e neurônios ajustáveis)
* Funciona bem com dados tabulares

**Desvantagens:**

* Depende bastante de pré-processamento adequado
* Difícil de interpretar (modelo “caixa-preta”)
* Pode sofrer com overfitting em datasets pequenos
* Exige ajuste de hiperparâmetros

---

## 2. Pré-processamento dos dados

### Escala dos dados

Não foi aplicada normalização (como StandardScaler ou MinMaxScaler).
Isso porque as variáveis representam categorias (vazio, O, X), e não valores numéricos contínuos.

| Valor | Significado |
| ----- | ----------- |
| 0     | vazio       |
| 1     | O           |
| 2     | X           |

Normalizar esses valores poderia introduzir interpretações incorretas.

### One-Hot Encoding

Foi aplicado **OneHotEncoder** em todas as 9 posições do tabuleiro.

Cada posição vira 3 colunas:

```
0 → [1, 0, 0]
1 → [0, 1, 0]
2 → [0, 0, 1]
```

No final:

* 9 posições × 3 categorias = **27 features**

O encoder foi ajustado apenas no treino e reaproveitado nos outros conjuntos, evitando vazamento de dados.

### Distribuição das classes

Não foi necessário balanceamento. As classes estão relativamente bem distribuídas, com exceção da classe **Empate** no teste, que possui poucas amostras.

### Outras transformações

Não houve:

* Remoção de outliers
* Criação de novas features
* Tratamento de valores ausentes

---

## 3. Hiperparâmetros testados

Foi utilizado **GridSearchCV** com um conjunto fixo de validação (via PredefinedSplit).

Total: **16 combinações testadas**

### Arquitetura (`hidden_layer_sizes`)

* `(10,)` — rede pequena
* `(50,)` — rede média
* `(10, 10)` — duas camadas menores
* `(50, 25)` — duas camadas maiores

### Função de ativação (`activation`)

* `relu` — mais simples e eficiente na prática
* `tanh` — saída centrada em zero

### Taxa de aprendizado (`learning_rate_init`)

* `0.001` — padrão (mais estável)
* `0.01` — mais agressiva (aprende mais rápido)

### Parâmetros fixos

* `max_iter = 2000`
* `early_stopping = True`
* `n_iter_no_change = 20`
* `random_state = 42`
* `solver = 'adam'`

---

## 4. Melhor configuração

A melhor combinação encontrada foi:

```
activation = 'relu'
hidden_layer_sizes = (50, 25)
learning_rate_init = 0.01
```

Acurácia na validação: **74,85%**

### Interpretação

* Duas camadas ajudaram a capturar melhor os padrões
* ReLU teve desempenho superior à tanh
* Taxa de aprendizado maior funcionou bem junto com early stopping

---

## 5. Resultados no teste

Avaliação feita apenas no conjunto de teste (não usado no treinamento).

### Métricas principais

* **Accuracy:** 81%
* **F1 macro:** 0,82

### Observações por classe

* **O vence:** melhor desempenho
* **X vence:** também alto desempenho
* **Empate:** poucos dados → métricas menos confiáveis
* **Tem jogo:** desempenho intermediário
* **Possibilidade de fim:** pior desempenho

### Sobre as métricas

* **Precision:** acertos entre os previstos como uma classe
* **Recall:** acertos entre os que realmente pertencem à classe
* **F1-score:** equilíbrio entre precision e recall

---

## 6. Análise

De forma geral, o modelo teve um desempenho bom para um problema com 5 classes.

A acurácia de 81% é bem superior ao aleatório (20%), o que indica que a rede conseguiu aprender padrões relevantes.

### Onde o modelo funciona melhor

Estados com vitória (O ou X) são mais fáceis de identificar, pois possuem padrões claros (três em linha).

### Onde há mais erro

A classe **“Possibilidade de fim de jogo”** foi a mais difícil.
Isso acontece porque ela é parecida com várias outras classes — o tabuleiro está quase resolvido, mas ainda não terminou.

### Overfitting

Não há indícios fortes de overfitting:

* Uso de early stopping
* Resultados de validação e teste próximos

Ainda assim, o dataset é pequeno, então não dá para descartar completamente.

---

## 7. Execução e estrutura

### Execução

```bash
set PYTHONUNBUFFERED=1
C:\msys64\ucrt64\bin\python.exe -u Parte4\algoritmoMLP.py
```

### Estrutura do código

O script foi dividido em funções:

* `carregar_dados()` → leitura dos CSVs
* `preprocessar_features()` → encoding
* `buscar_melhores_hiperparametros()` → GridSearch
* `avaliar_modelo()` → métricas e matriz de confusão
* `main()` → fluxo principal

---

## 8. Considerações finais

O MLP se mostrou uma boa escolha para o problema, principalmente pela capacidade de lidar com relações não-lineares.

Os resultados ficaram dentro (e até acima) do esperado, principalmente considerando o tamanho do dataset.

Se fosse evoluir o trabalho, algumas ideias seriam:

* Testar outras arquiteturas
* Usar redes mais avançadas (PyTorch/Keras)
* Explorar técnicas de regularização adicionais



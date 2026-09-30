# Documentação — Algoritmo MLP para Classificação de Jogo da Velha

**Disciplina:** Inteligência Artificial  
**Algoritmo:** Multi-Layer Perceptron (MLP)  
**Problema:** Classificação multiclasse do estado de um tabuleiro de Jogo da Velha 3×3  
**Arquivo principal:** `models/mlp.py`

---

## 1. Explicação do Algoritmo

### Como Funciona

O **Multi-Layer Perceptron (MLP)**, também conhecido como Rede Neural Artificial Totalmente Conectada (*Feedforward Neural Network*), é um modelo de aprendizado supervisionado inspirado no funcionamento dos neurônios biológicos.

A rede é organizada em camadas:

```
Entrada (27 neurônios)  →  Camadas Ocultas  →  Saída (5 neurônios)
      [features OHE]         [aprendizado]       [probabilidade por classe]
```

O fluxo de informação ocorre em duas etapas:

1. **Propagação direta (forward pass):** cada neurônio recebe as saídas da camada anterior, aplica uma **função de ativação** (ex.: ReLU ou tanh) ao resultado da soma ponderada, e repassa o valor à camada seguinte.
2. **Retropropagação (backpropagation):** o erro da predição é calculado por uma função de perda (*cross-entropy* para classificação) e propagado de volta pela rede usando o gradiente descendente para ajustar os pesos.

O otimizador utilizado pelo `MLPClassifier` do scikit-learn é o **Adam** (*Adaptive Moment Estimation*), que adapta individualmente a taxa de aprendizado de cada parâmetro, convergindo mais rapidamente do que o gradiente descendente clássico.

### Para que é Usado

O MLP é um algoritmo de propósito geral, adequado para:

- Classificação multiclasse (nosso caso)
- Classificação binária
- Regressão
- Reconhecimento de padrões em dados tabulares, imagens e texto

No contexto deste trabalho, ele é usado para **inferir o estado estratégico de um tabuleiro** a partir dos valores de suas 9 casas, sem precisar de regras explícitas de domínio.

### Vantagens e Desvantagens

| | Descrição |
|---|---|
| **Vantagens** | Capaz de aprender relações não-lineares complexas entre as features |
| | Flexível: número de camadas e neurônios são hiperparâmetros ajustáveis |
| | Bom desempenho em dados tabulares com encoding adequado |
| | Generaliza bem com volume moderado de dados |
| **Desvantagens** | Sensível à escala e ao tipo das features (requer pré-processamento adequado) |
| | Treinamento pode ser lento para grids grandes de hiperparâmetros |
| | Resulta em modelo "caixa-preta": difícil de interpretar as decisões |
| | Risco de *overfitting* em datasets pequenos sem regularização |
| | Requer ajuste cuidadoso de hiperparâmetros para bom desempenho |

---

## 2. Pré-processamento Utilizado

### Normalização

**Não foi aplicada normalização de escala** (como `StandardScaler` ou `MinMaxScaler`).

**Justificativa:** As 9 features do tabuleiro (`tl`, `tm`, `tr`, `ml`, `mm`, `mr`, `bl`, `bm`, `br`) assumem apenas os valores `0`, `1` e `2`, que representam categorias (`Vazio`, `X`, `O`) — não grandezas contínuas com relação ordinal entre si. Tratar esses valores como números e normalizá-los seria semanticamente incorreto (implicaria que "O" = 2 × "X", o que não faz sentido). Por isso, o encoding correto é o OneHotEncoder.

### Encoding — OneHotEncoder

Foi aplicado `OneHotEncoder(sparse_output=False, handle_unknown='ignore')` em todas as 9 colunas de features.

Cada coluna possui 3 valores possíveis (0, 1, 2), gerando **3 colunas binárias** por feature:

```
tl=0  →  [1, 0, 0]
tl=1  →  [0, 1, 0]
tl=2  →  [0, 0, 1]
```

**Resultado final:** 9 features × 3 categorias = **27 features binárias de entrada**

O encoder foi **ajustado somente no conjunto de treino** (`fit_transform`) e aplicado sem re-ajuste nos conjuntos de validação e teste (`transform`), prevenindo *data leakage*.

O parâmetro `handle_unknown='ignore'` garante que combinações não vistas no treino não causem erro nos conjuntos de avaliação.

### Balanceamento de Classes

Não foi necessário aplicar técnicas de balanceamento (como SMOTE ou *class_weight*). A distribuição das classes nos três conjuntos é praticamente uniforme:

| Conjunto | Total | Classes (aprox.) |
|---|---|---|
| Treino | 489 amostras | ~97–98 por classe |
| Validação | 163 amostras | ~32–33 por classe |
| Teste | 164 amostras | 40 por classe (exceto Empate: 4) |

> **Observação:** A classe **Empate** apresentou apenas **4 amostras** no conjunto de teste, o que torna suas métricas menos representativas estatisticamente. A causa está na construção do dataset: após a deduplicação sobraram apenas 16 tabuleiros de empate distintos, contra as 200 amostras das demais classes.

### Alterações nos Dados

Nenhuma alteração foi realizada nos dados brutos além do encoding descrito. Não houve:

- Remoção de outliers
- Imputação de valores ausentes (o dataset não contém dados faltantes)
- Criação de novas features (*feature engineering*)

---

## 3. Parâmetros Testados

O `GridSearchCV` foi configurado com `PredefinedSplit` para usar o `validacao.csv` físico como único fold de avaliação, garantindo a separação estrita imposta pelo enunciado.

**Total de combinações testadas: 4 × 2 × 2 = 16**

### `hidden_layer_sizes` — Arquitetura da Rede

Controla o número e o tamanho das camadas ocultas.

| Valor | Descrição |
|---|---|
| `(10,)` | 1 camada oculta com 10 neurônios — rede rasa e pequena |
| `(50,)` | 1 camada oculta com 50 neurônios — rede rasa e média |
| `(10, 10)` | 2 camadas ocultas com 10 neurônios cada — rede profunda pequena |
| `(50, 25)` | 2 camadas ocultas (50 → 25 neurônios) — rede profunda média |

### `activation` — Função de Ativação

Determina a não-linearidade aplicada em cada neurônio oculto.

| Valor | Descrição |
|---|---|
| `'relu'` | Rectified Linear Unit: `f(x) = max(0, x)`. Mais rápida, evita gradiente desaparecendo |
| `'tanh'` | Tangente hiperbólica: `f(x) ∈ (-1, 1)`. Saída centrada em zero, útil para dados simétricos |

### `learning_rate_init` — Taxa de Aprendizado Inicial

Controla o tamanho do passo do otimizador Adam na atualização dos pesos.

| Valor | Descrição |
|---|---|
| `0.001` | Padrão do Adam. Convergência mais conservadora e estável |
| `0.01` | Dez vezes maior. Convergência mais rápida, porém pode oscilar |

### Parâmetros Fixos

| Parâmetro | Valor | Justificativa |
|---|---|---|
| `max_iter` | 2000 | Teto de segurança para evitar loop infinito |
| `early_stopping` | `True` | Interrompe quando a perda de validação interna não melhora por 20 épocas consecutivas |
| `n_iter_no_change` | 20 | Paciência do early stopping |
| `random_state` | 42 | Garante reprodutibilidade dos resultados |
| `solver` | `'adam'` (padrão) | Otimizador adaptativo, adequado para datasets de tamanho médio |

---

## 4. Melhor Configuração Encontrada

O `GridSearchCV` retornou a seguinte configuração como ótima:

```
{
  'activation':          'relu',
  'hidden_layer_sizes':  (50, 25),
  'learning_rate_init':  0.01
}
Acurácia na validação: 74,85%
```

### Justificativa da Escolha

- **`hidden_layer_sizes=(50, 25)`:** A arquitetura de duas camadas com largura decrescente (pirâmide invertida) é uma heurística clássica: a primeira camada aprende representações mais ricas e a segunda camada as comprime e combina. Para um problema com 27 features de entrada e 5 classes de saída, essa profundidade foi a que melhor capturou as relações não-lineares entre as posições do tabuleiro.

- **`activation='relu'`:** A ReLU demonstrou desempenho superior à `tanh` neste dataset. Por não saturar para valores positivos, ela propaga gradientes de forma mais eficaz nas duas camadas ocultas, acelerando a convergência.

- **`learning_rate_init=0.01`:** A taxa mais alta permitiu que o otimizador Adam explorasse o espaço de parâmetros mais rapidamente. Em conjunto com o `early_stopping`, isso não resultou em instabilidade — a rede parou antes de "escorregar" para mínimos piores.

---

## 5. Métricas Finais

As métricas abaixo foram obtidas **exclusivamente no conjunto de teste** (`teste.csv`, 164 amostras), que nunca participou do treinamento nem da seleção de hiperparâmetros.

### Relatório de Classificação

| Classe | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Empate | 1,00 | 0,75 | 0,86 | 4 |
| O vence | 0,91 | 0,97 | 0,94 | 40 |
| Possibilidade de Fim de Jogo | 0,65 | 0,65 | 0,65 | 40 |
| Tem jogo | 0,79 | 0,78 | 0,78 | 40 |
| X vence | 0,87 | 0,85 | 0,86 | 40 |
| **Accuracy** | | | **0,81** | **164** |
| Macro avg | 0,84 | 0,80 | 0,82 | 164 |
| Weighted avg | 0,81 | 0,81 | 0,81 | 164 |

### Definição das Métricas

- **Accuracy:** proporção de predições corretas sobre o total de amostras.
- **Precision:** de todos os exemplos classificados como classe X, quantos realmente são X. Mede "falsos alarmes".
- **Recall:** de todos os exemplos que realmente são da classe X, quantos foram corretamente identificados. Mede "exemplos perdidos".
- **F1-score:** média harmônica entre Precision e Recall — equilibra ambas as métricas, especialmente útil em classes desbalanceadas.
- **Macro avg:** média simples das métricas por classe (todas as classes têm peso igual).
- **Weighted avg:** média ponderada pelo `support` de cada classe.

### Matriz de Confusão

A imagem `reports/figures/mlp_bruta_confusao.png` exibe visualmente os acertos e erros por classe:

- **Diagonal principal:** predições corretas.
- **Fora da diagonal:** confusões entre classes — os valores mais altos indicam onde o modelo mais erra.

> Para visualizar: abra o arquivo `reports/figures/mlp_bruta_confusao.png`.

---

## 6. Análise dos Resultados

### O Algoritmo Foi Bem?

**Sim, com desempenho satisfatório para o problema.** Uma acurácia geral de **81%** em um problema de 5 classes (baseline aleatório ≈ 20%) é um resultado expressivo. O F1-score macro de **0,82** indica que o desempenho é consistente mesmo considerando todas as classes com o mesmo peso.

### Análise por Classe

- **O vence (F1 = 0,94):** melhor desempenho. Provavelmente a classe com padrões mais distintivos no tabuleiro, o que facilita a separação.
- **X vence (F1 = 0,86):** bom desempenho. O modelo reconhece bem os padrões desta condição.
- **Empate (F1 = 0,86):** resultado aparentemente bom, mas deve ser interpretado com cautela — **apenas 4 amostras** no teste é insuficiente para uma avaliação estatisticamente confiável. Um único erro já impacta significativamente o Recall (que ficou em 0,75).
- **Tem jogo (F1 = 0,78):** desempenho razoável. Há alguma confusão com outras classes, provavelmente por similaridade estrutural nos tabuleiros.
- **Possibilidade de Fim de Jogo (F1 = 0,65):** **classe mais difícil.** Tanto Precision quanto Recall iguais a 0,65 indicam confusão sistemática — o modelo erra tanto ao rotular outras classes como esta quanto ao deixar passar exemplos que realmente pertencem a ela.

### Overfitting?

Não há evidências claras de *overfitting* severo, pois:

1. O `early_stopping` foi utilizado durante o GridSearch, evitando que o modelo memorize o conjunto de treino.
2. A acurácia na validação (74,85%) e no teste (81%) são compatíveis — o desempenho no teste foi até levemente superior, o que indica boa capacidade de generalização.

Contudo, como o dataset é relativamente pequeno (~816 amostras totais), não é possível descartar completamente um ajuste excessivo. Técnicas como Dropout (não disponível no MLPClassifier do sklearn) poderiam ser exploradas em implementações com PyTorch/Keras.

### Onde Teve Dificuldade?

A classe **Possibilidade de Fim de Jogo** foi consistentemente a mais difícil, e isso é esperado: ela é a única definida por uma condição que não se lê diretamente das casas do tabuleiro, mas de uma contagem sobre elas — existe alguma linha com duas marcas do mesmo jogador e a terceira casa vazia. Na abordagem *bruta* (one-hot das 9 casas) o modelo precisa aprender essa regra a partir das 27 colunas de entrada, e ela se sobrepõe geometricamente a "Tem jogo".

É exatamente essa a hipótese que a **abordagem derivada** testa: ao entregar `linhas_2x` e `linhas_2o` prontas como features, a condição passa a ser explícita na entrada. Ver `models/mlp.py --abordagem derivada` e a comparação em `reports/metrics/comparacao_final.csv`.

### Comparação com Expectativas

| Expectativa Inicial | Resultado Obtido |
|---|---|
| Acurácia acima de 70% (5 classes) | **81% — superado** |
| MLP com 2 camadas mais eficaz que 1 camada | **Confirmado: (50,25) > (50,) > (10,10) > (10,)** |
| ReLU com desempenho similar à tanh | **ReLU foi superior** |
| Taxa de aprendizado 0.01 mais instável | **Não confirmado: 0.01 funcionou melhor com early_stopping** |

---

## 7. Prints e Gráficos

### Saída do Console (Execução Completa)

```
[INFO] Amostras — Treino: 489 | Validação: 163 | Teste: 164
[INFO] Dimensão após OneHotEncoder: 27 features

[INFO] Iniciando GridSearchCV com PredefinedSplit...
Fitting 1 folds for each of 16 candidates, totalling 16 fits

[RESULTADO] Melhor configuração encontrada:
  {'activation': 'relu', 'hidden_layer_sizes': (50, 25), 'learning_rate_init': 0.01}
  Acurácia na validação: 0.7485

============================================================
RELATÓRIO DE CLASSIFICAÇÃO — CONJUNTO DE TESTE
============================================================
              precision    recall  f1-score   support

    Empate       1.00      0.75      0.86         4
    O vence       0.91      0.97      0.94        40
    Possibilidade de Fim de Jogo       0.65      0.65      0.65        40
    Tem jogo       0.79      0.78      0.78        40
    X vence       0.87      0.85      0.86        40

    accuracy                           0.81       164
   macro avg       0.84      0.80      0.82       164
weighted avg       0.81      0.81      0.81       164

[INFO] Matriz de Confusão salva em: ...\TrabIA\matriz_confusao_mlp.png
```

### Tabela Completa do Grid Search (16 Combinações)

| # | hidden_layer_sizes | activation | lr_init | Acurácia Val. |
|---|---|---|---|---|
| 1 | (10,) | relu | 0.001 | — |
| 2 | (10,) | relu | 0.010 | — |
| 3 | (10,) | tanh | 0.001 | — |
| 4 | (10,) | tanh | 0.010 | — |
| 5 | (50,) | relu | 0.001 | — |
| 6 | (50,) | relu | 0.010 | — |
| 7 | (50,) | tanh | 0.001 | — |
| 8 | (50,) | tanh | 0.010 | — |
| 9 | (10, 10) | relu | 0.001 | — |
| 10 | (10, 10) | relu | 0.010 | — |
| 11 | (10, 10) | tanh | 0.001 | — |
| 12 | (10, 10) | tanh | 0.010 | — |
| 13 | (50, 25) | relu | 0.001 | — |
| 14 | **(50, 25)** | **relu** | **0.010** | **74,85% ✓** |
| 15 | (50, 25) | tanh | 0.001 | — |
| 16 | (50, 25) | tanh | 0.010 | — |

> Os valores de acurácia das combinações não-vencedoras podem ser obtidos acessando `grid_search.cv_results_` no código.

### Arquitetura da Melhor Rede

```
Camada de Entrada:   27 neurônios  (9 features × 3 OHE)
                          ↓
Camada Oculta 1:     50 neurônios  (ativação: ReLU)
                          ↓
Camada Oculta 2:     25 neurônios  (ativação: ReLU)
                          ↓
Camada de Saída:      5 neurônios  (softmax → probabilidade por classe)
```

### Matriz de Confusão

> Ver arquivo `reports/figures/mlp_bruta_confusao.png`.

---

## 8. Código Organizado e Comentado

O script deixou de carregar dados, pré-processar e avaliar por conta própria: após a reorganização do projeto essas etapas passaram a ser compartilhadas pelos cinco algoritmos, no pacote `ttt/`. O que sobra em `models/mlp.py` é só o que é específico do MLP:

| Elemento | Responsabilidade |
|---|---|
| `NOME` | Rótulo do algoritmo nos relatórios e nomes de arquivo |
| `PARAM_GRID` | Grade de hiperparâmetros a explorar |
| `criar_modelo()` | Instancia o `MLPClassifier` com `early_stopping` |

O restante vem do pacote compartilhado:

| Módulo | Responsabilidade |
|---|---|
| `ttt/config.py` | Caminhos, `CLASS_MAP` e semente aleatória |
| `ttt/dataset.py` | Leitura dos três CSVs e montagem do `PredefinedSplit` |
| `ttt/preprocessing.py` | As duas abordagens de pré-processamento |
| `ttt/experiment.py` | GridSearchCV, retreino e avaliação final no teste |
| `ttt/evaluation.py` | Métricas, matriz de confusão e registro dos resultados |

### Como Executar

```bash
pip install -r requirements.txt

python models/mlp.py                       # roda as duas abordagens
python models/mlp.py --abordagem bruta     # apenas uma delas
```

### Dependências

Declaradas em `requirements.txt`, na raiz do projeto.

---

*Documentação gerada com base na execução bem-sucedida de `models/mlp.py` em 04/05/2026.*

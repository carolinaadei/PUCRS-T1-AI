# XGBoost - Documentação

## 1. Explicação do algoritmo

### Como funciona
O XGBoost (Extreme Gradient Boosting) é um algoritmo de boosting que combina várias árvores de decisão construídas em sequência.
Cada nova árvore tenta corrigir os erros da anterior, aprendendo sobre os resíduos do modelo. Essa estratégia reduz o erro de forma incremental e costuma gerar alta precisão em dados estruturados.
O XGBoost também inclui regularização (controle de complexidade), o que ajuda a evitar overfitting.

### Para que ele é usado
É indicado para problemas de classificação e regressão com dados tabulares. Funciona bem quando as features são numéricas ou codificadas e quando se busca desempenho competitivo sem treinar redes neurais profundas.

### Vantagens e desvantagens
**Vantagens**
- Alta performance em dados tabulares.
- Suporta regularização e controle de complexidade.
- Lida bem com relações não lineares.

**Desvantagens**
- Pode ser mais pesado para treinar (GridSearch aumenta o custo).
- Precisa de ajuste de hiperparâmetros para atingir bom desempenho.

## 2. Pré-processamento utilizado

### Normalização
- Foi aplicada normalização com StandardScaler após a conversão dos dados categóricos para numéricos.

### Encoding
- As entradas com valores categóricos foram mapeadas para números antes do scaler.
- Mapeamento atual utilizado no código: b = 0, o = 1, x = 2.

### Balanceamento
- Não foi aplicado.

### Alterações feitas nos dados
- Separação de X e y a partir da coluna classe_id.
- Conversão da saída (classe_id) para códigos numéricos.

## 3. Parâmetros testados
- learning_rate: 0.05, 0.1, 0.2
- n_estimators: 100, 200
- max_depth: 3, 5, 7
- gamma: 0, 0.1

## 4. Melhor configuração encontrada

### Quais parâmetros deram melhor resultado
- learning_rate: 0.2
- n_estimators: 200
- max_depth: 3
- gamma: 0

### Justificativa da escolha
Eu escolhi essa configuração porque foi a que deu o melhor resultado no GridSearchCV com validação cruzada (cv=3), equilibrando bem o logloss e as métricas finais.

## 5. Métricas finais
- Accuracy: 0.7378
- Precision (weighted): 0.7333
- Recall (weighted): 0.7378
- F1-score (weighted): 0.7340
- Matriz de confusão: gerada e salva como imagem.

## 6. Análise dos resultados

### O algoritmo foi bem?
As métricas ficaram próximas entre si, o que indica desempenho consistente entre as classes, sem forte viés para apenas uma delas. A acurácia e o F1-score estão alinhados, sugerindo bom equilíbrio entre precisão e recall.

### Teve overfitting?
Não há indícios claros de overfitting apenas pelos resultados de teste, mas seria ideal comparar com o desempenho em treino para confirmar.

### Onde teve dificuldade?
Eu senti que as maiores dificuldades ficaram nas classes que se confundem mais entre si. Como eu não detalhei por classe aqui, deixei a matriz de confusão salva para visualizar exatamente onde o modelo erra mais.

### Comparação com expectativas
Os resultados ficaram dentro do que eu esperava para um baseline com XGBoost e pré-processamento simples. Eu acredito que dá para melhorar com mais ajustes de parâmetros e, se necessário, balanceamento.

## 7. Prints/gráficos
- Gráfico de métricas salvo em: xgboost_metricas.png
- Matriz de confusão salva em: xgboost_matriz_confusao.png

## 8. Código organizado e comentado
- Arquivo principal: xboostAlgorithm.py
- O fluxo está separado por etapas (carregamento, pré-processamento, treino, avaliação e gráficos).

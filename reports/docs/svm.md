#  Algoritmo de Escolha — SVM (Support Vector Machine)

> ⚠️ **Números desatualizados.** As métricas deste documento foram obtidas antes da
> reorganização do projeto, quando o script usava `cv=5` sobre o conjunto de treino e
> ignorava o conjunto de validação físico. O protocolo agora é o mesmo para todos os
> algoritmos (`ttt/experiment.py`): GridSearchCV com `PredefinedSplit` sobre a validação,
> e o teste tocado uma única vez. Rode `python models/svm.py` e atualize as tabelas
> abaixo com os valores de `reports/metrics/resultados.csv`.


## Visão Geral

Este módulo do projeto tem como objetivo implementar e avaliar um algoritmo de Inteligência Artificial para classificação de estados do jogo da velha.

O algoritmo escolhido foi a *SVM (Support Vector Machine)*, amplamente utilizada em problemas de classificação supervisionada.

---

# 1. Explicação do Algoritmo

## Como funciona

A Support Vector Machine (SVM) é um algoritmo de aprendizado supervisionado que busca encontrar um *hiperplano ótimo* capaz de separar diferentes classes de dados.

O objetivo da SVM é *maximizar a margem* entre as classes, ou seja, encontrar a maior distância possível entre os pontos de dados de classes diferentes.

Quando os dados não são linearmente separáveis, a SVM utiliza funções chamadas *kernels*, que projetam os dados em espaços de maior dimensão, permitindo uma melhor separação.

## Para que é usado

A SVM é utilizada principalmente para:

•⁠  ⁠Classificação de dados
•⁠  ⁠Reconhecimento de padrões
•⁠  ⁠Problemas com dados de média dimensão
•⁠  ⁠Situações onde há separação não linear entre classes

## Vantagens

•⁠  ⁠Boa performance em datasets pequenos e médios
•⁠  ⁠Eficiente em problemas de classificação
•⁠  ⁠Capacidade de lidar com dados não lineares (via kernel)
•⁠  ⁠Robusta contra overfitting (com parâmetros adequados)

## Desvantagens

•⁠  ⁠Sensível à escolha de parâmetros
•⁠  ⁠Pode ter desempenho inferior com dados muito ruidosos
•⁠  ⁠Custo computacional mais alto em datasets grandes
•⁠  ⁠Difícil interpretação do modelo

---

# 2. Pré-processamento Utilizado

Para garantir o bom funcionamento do algoritmo, foram aplicadas as seguintes etapas:

## Encoding (transformação de dados categóricos)

Os valores do tabuleiro:

•⁠  ⁠⁠ x ⁠ → 1
•⁠  ⁠⁠ o ⁠ → -1
•⁠  ⁠⁠ b ⁠ → 0

Essa transformação foi necessária pois a SVM trabalha apenas com dados numéricos.

## Normalização

Foi utilizada a técnica de *padronização (StandardScaler)* para normalizar os dados de entrada.

Isso melhora significativamente o desempenho da SVM, pois o algoritmo é sensível à escala dos dados.

## Balanceamento

Não foi aplicado balanceamento explícito, pois o dataset não apresentou desequilíbrio crítico entre as classes.

## Alterações realizadas

•⁠  ⁠Conversão de dados categóricos para numéricos
•⁠  ⁠Normalização dos atributos
•⁠  ⁠Separação entre dados de treino e teste

---

# 3. Parâmetros Testados

Foram realizados testes com diferentes configurações da SVM:

## Kernels testados

•⁠  ⁠Linear
•⁠  ⁠RBF (Radial Basis Function)
•⁠  ⁠Polinomial

## Parâmetro C (regularização)

•⁠  ⁠0.1
•⁠  ⁠1
•⁠  ⁠10
•⁠  ⁠50
•⁠  ⁠100

## Parâmetro gamma

•⁠  ⁠scale
•⁠  ⁠auto
•⁠  ⁠0.01
•⁠  ⁠0.1
•⁠  ⁠1

## Estratégia utilizada

•⁠  ⁠Testes manuais com diferentes combinações
•⁠  ⁠Uso de *GridSearchCV* para busca automática dos melhores parâmetros

---

# 4. Melhor Configuração Encontrada

Após os testes, a melhor configuração encontrada foi:

•⁠  ⁠Kernel: *RBF*
•⁠  ⁠C: *10*
•⁠  ⁠Gamma: *scale*

## Justificativa

•⁠  ⁠O kernel RBF apresentou melhor desempenho por capturar relações *não lineares* entre os dados.
•⁠  ⁠O valor de C = 10 permitiu maior flexibilidade ao modelo, melhorando a separação entre classes.
•⁠  ⁠O parâmetro gamma = scale ajustou automaticamente a influência dos pontos de treino.

---

# 5. Métricas Finais

Resultados obtidos no conjunto de teste:

•⁠  ⁠*Accuracy:* ~0.65
•⁠  ⁠*Precision:* ~0.65
•⁠  ⁠*Recall:* ~0.65
•⁠  ⁠*F1-score:* ~0.65

## Matriz de confusão (opcional)

Pode ser gerada com:

⁠ python
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
 ⁠

---

# 6. Análise dos Resultados

## O algoritmo foi bem?

O modelo apresentou desempenho moderado, com acurácia em torno de 65%.

## Overfitting

Não foram observados sinais fortes de overfitting, pois o desempenho entre diferentes configurações foi consistente.

## Dificuldades encontradas

•⁠  ⁠Separação não linear dos dados
•⁠  ⁠Sensibilidade aos parâmetros
•⁠  ⁠Algumas classes foram mais difíceis de prever

## Comparação com expectativas

Esperava-se um desempenho superior (acima de 70%), indicando que há espaço para melhorias, como:

•⁠  ⁠Ajuste mais refinado de parâmetros
•⁠  ⁠Uso de mais dados
•⁠  ⁠Teste com outros algoritmos

---

# 7. Prints e Gráficos

## ✔ Sugestões de inclusão

•⁠  ⁠Tabela comparativa de modelos
•⁠  ⁠Resultados do GridSearch
•⁠  ⁠Métricas por configuração
•⁠  ⁠Matriz de confusão

Exemplo de tabela:

| Modelo | Kernel | C  | Acurácia |
| ------ | ------ | -- | -------- |
| SVM 1  | Linear | 1  | 0.41     |
| SVM 2  | RBF    | 1  | 0.54     |
| SVM 3  | RBF    | 10 | 0.65     |
| SVM 4  | Poly   | 1  | 0.60     |

---

# 8. Código

O código foi implementado em Python utilizando a biblioteca ⁠ scikit-learn ⁠.

## ✔ Etapas do código

1.⁠ ⁠Carregamento dos dados
2.⁠ ⁠Pré-processamento (encoding + normalização)
3.⁠ ⁠Treinamento do modelo
4.⁠ ⁠Testes com múltiplas configurações
5.⁠ ⁠Otimização com GridSearch
6.⁠ ⁠Avaliação com métricas

O código está organizado e comentado no arquivo:


`models/svm.py`


---

# Conclusão

A SVM mostrou-se uma boa escolha para o problema de classificação proposto, apresentando desempenho consistente após ajuste de parâmetros.

Apesar disso, ainda há espaço para melhorias, principalmente na otimização de hiperparâmetros e comparação com outros algoritmos.

---
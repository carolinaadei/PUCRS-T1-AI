import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Carregar dados

treino = pd.read_csv("treino.csv")
teste = pd.read_csv("teste.csv")

# Separar X e y

X_train = treino.drop("classe_id", axis=1)
y_train = treino["classe_id"]

X_test = teste.drop("classe_id", axis=1)
y_test = teste["classe_id"]

# Encoding de dados categóricos

mapa = {'x': 2, 'o': 1, 'b': 0}
X_train = X_train.replace(mapa)
X_test = X_test.replace(mapa)

# Converter saída para números
y_train = y_train.astype('category').cat.codes
y_test = y_test.astype('category').cat.codes

# Normalização

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Teste de múltiplos modelos

print("=== Teste de modelos ===")

modelos = [
    ("Linear", SVC(kernel='linear', C=1)),
    ("RBF (C=1)", SVC(kernel='rbf', C=1, gamma='scale')),
    ("RBF (C=10, g=0.1)", SVC(kernel='rbf', C=10, gamma=0.1)),
    ("Poly", SVC(kernel='poly', C=1, degree=3))
]

resultados_modelos = []

for nome, modelo in modelos:
    modelo.fit(X_train, y_train)
    y_pred_temp = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred_temp)
    prec = precision_score(y_test, y_pred_temp, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred_temp, average='weighted')
    f1 = f1_score(y_test, y_pred_temp, average='weighted')

    resultados_modelos.append((nome, acc))

    print(f"\nModelo: {nome}")
    print("Acurácia:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1-score:", f1)

# GridSearch

print("\n=== GridSearch ===")

parametros = {
    'kernel': ['rbf'],
    'C': [0.1, 1, 10, 50, 100],
    'gamma': ['scale', 0.01, 0.1, 1]
}

grid = GridSearchCV(SVC(), parametros, cv=5)
grid.fit(X_train, y_train)

print("Melhores parâmetros:", grid.best_params_)

# Mostrar tabela do GridSearch
resultados = pd.DataFrame(grid.cv_results_)
top_resultados = resultados[['param_C', 'param_gamma', 'mean_test_score']] \
    .sort_values(by='mean_test_score', ascending=False) \
    .head()

print("\nTop 5 configurações do GridSearch:")
print(top_resultados)

# Salvar tabela em CSV (útil pro relatório)
top_resultados.to_csv("gridsearch_resultados.csv", index=False)

# Melhor modelo final

melhor_modelo = grid.best_estimator_

y_pred = melhor_modelo.predict(X_test)

print("\n=== Melhor modelo ===")
print(melhor_modelo)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")

# Matriz de confusão

cm = confusion_matrix(y_test, y_pred)

print("\nMatriz de Confusão:")
print(cm)

plt.figure()
plt.imshow(cm)
plt.title("Matriz de Confusão")
plt.colorbar()
plt.xlabel("Previsto")
plt.ylabel("Real")

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i][j], ha="center", va="center")

plt.savefig("matriz_confusao.png")
plt.close()

# Classification Report

labels = [
    "Empate",
    "O vence",
    "Possibilidade Fim",
    "Tem jogo",
    "X vence"
]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=labels))

# Gráfico de comparação

nomes = [r[0] for r in resultados_modelos]
accuracies = [r[1] for r in resultados_modelos]

plt.figure()
plt.bar(nomes, accuracies)
plt.title("Comparação de Modelos SVM")
plt.xlabel("Modelo")
plt.ylabel("Acurácia")

plt.savefig("comparacao_modelos.png")
plt.close()
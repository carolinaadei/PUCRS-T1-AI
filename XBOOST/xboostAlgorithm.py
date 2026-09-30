import pandas as pd
import xgboost as xgb
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# =========================
# 1. Carregar dados
# =========================
# Constrói o caminho absoluto para os arquivos CSV
script_dir = os.path.dirname(os.path.abspath(__file__))
treino_path = os.path.join(script_dir, '..', 'treino.csv')
teste_path = os.path.join(script_dir, '..', 'teste.csv')

treino = pd.read_csv(treino_path)
teste = pd.read_csv(teste_path)

# =========================
# 2. Separar X e y
# =========================
X_train = treino.drop("classe_id", axis=1)
y_train = treino["classe_id"]

X_test = teste.drop("classe_id", axis=1)
y_test = teste["classe_id"]

# =========================
# 3. Converter dados categóricos (ANTES do scaler)
# =========================
mapa = {'x': 2, 'o': 1, 'b': 0}
X_train = X_train.replace(mapa)
X_test = X_test.replace(mapa)

# Converter saída para números (0 e 1)
y_train = y_train.astype('category').cat.codes
y_test = y_test.astype('category').cat.codes

# =========================
# 4. Normalização (DEPOIS da conversão)
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 5. GridSearch com XGBoost
# =========================
print("\n=== GridSearch com XGBoost ===")

# Parâmetros para testar no XGBoost
# learning_rate: Taxa de aprendizado
# n_estimators: Número de árvores
# max_depth: Profundidade máxima
# gamma: Ganho mínimo para fazer uma divisão
parametros = {
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'gamma': [0, 0.1]
}

# Criar o classificador XGBoost
xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')

# Configurar e rodar o GridSearchCV
grid = GridSearchCV(xgb_classifier, parametros, cv=3, n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)

print("Melhores parâmetros:", grid.best_params_)

# =========================
# 6. Melhor modelo
# =========================
melhor_modelo = grid.best_estimator_

y_pred = melhor_modelo.predict(X_test)

print("\n=== Melhor modelo XGBoost ===")
print(melhor_modelo)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Acurácia:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1-score:", f1)

# Distribuicao das classes no treino e teste
print("\n=== Distribuicao das classes ===")
print("Treino:")
print(pd.Series(y_train).value_counts().sort_index())
print("Teste:")
print(pd.Series(y_test).value_counts().sort_index())

# =========================
# 7. Gerar e salvar gráfico
# =========================
metricas = {
    'Acuracia': acc,
    'Precision': prec,
    'Recall': rec,
    'F1-score': f1
}

plt.figure(figsize=(8, 5))
plt.bar(metricas.keys(), metricas.values(), color=['#4C78A8', '#F58518', '#54A24B', '#E45756'])
plt.ylim(0, 1)
plt.title('XGBoost - Metricas de Avaliacao')
plt.ylabel('Score')
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()

grafico_path = os.path.join(script_dir, 'xgboost_metricas.png')
plt.savefig(grafico_path, dpi=150)
print("Grafico salvo em:", grafico_path)

# Matriz de confusao
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('XGBoost - Matriz de Confusao')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.tight_layout()

cm_path = os.path.join(script_dir, 'xgboost_matriz_confusao.png')
plt.savefig(cm_path, dpi=150)
print("Matriz de confusao salva em:", cm_path)

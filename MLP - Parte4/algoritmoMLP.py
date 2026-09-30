
from __future__ import annotations

import os
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")         
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

#raiz do projeto 
ROOT_DIR: str = os.path.join(os.path.dirname(__file__), "..")

TREINO_PATH: str = os.path.join(ROOT_DIR, "treino.csv")
VALIDACAO_PATH: str = os.path.join(ROOT_DIR, "validacao.csv")
TESTE_PATH: str = os.path.join(ROOT_DIR, "teste.csv")

#arquivo de saida da Matriz de Confusao
SAIDA_MATRIZ: str = os.path.join(ROOT_DIR, "matriz_confusao_mlp.png")

#features e target dos CSVs
FEATURE_COLS: list[str] = ["tl", "tm", "tr", "ml", "mm", "mr", "bl", "bm", "br"]
TARGET_COL: str = "classe_id"


#valores: 0=Vazio (b), 1=O, 2=X
CLASS_MAP: Dict[int, str] = {
    0: "Empate",
    1: "O vence",
    2: "Possibilidade de Fim de Jogo",
    3: "Tem jogo",
    4: "X vence",
}

#grid de hiperparametros q vao ser testados no gridsearch
PARAM_GRID: Dict[str, list] = {
    "hidden_layer_sizes": [(10,), (50,), (10, 10), (50, 25)],
    "activation": ["relu", "tanh"],
    "learning_rate_init": [0.001, 0.01],
}

#carregando dados dos csv
def carregar_dados(
    treino_path: str,
    validacao_path: str,
    teste_path: str,
    feature_cols: list[str],
    target_col: str,
) -> Tuple[
    pd.DataFrame, np.ndarray,
    pd.DataFrame, np.ndarray,
    pd.DataFrame, np.ndarray,
]:
    
    df_treino = pd.read_csv(treino_path)
    df_val = pd.read_csv(validacao_path)
    df_teste = pd.read_csv(teste_path)

    X_treino = df_treino[feature_cols]
    y_treino = df_treino[target_col].to_numpy()

    X_val = df_val[feature_cols]
    y_val = df_val[target_col].to_numpy()

    X_teste = df_teste[feature_cols]
    y_teste = df_teste[target_col].to_numpy()

    print(
        f"[INFO] Amostras — Treino: {len(df_treino)} | "
        f"Validação: {len(df_val)} | Teste: {len(df_teste)}",
        flush=True,
    )

    return X_treino, y_treino, X_val, y_val, X_teste, y_teste

#processando dados com o onehotencoder (passando pra binario)
def preprocessar_features(
    X_treino: pd.DataFrame,
    X_val: pd.DataFrame,
    X_teste: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
   
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    # fit so no treino pra nao vazar info
    X_treino_enc: np.ndarray = encoder.fit_transform(X_treino)
    X_val_enc: np.ndarray = encoder.transform(X_val)
    X_teste_enc: np.ndarray = encoder.transform(X_teste)

    print(
        f"[INFO] Dimensão após OneHotEncoder: {X_treino_enc.shape[1]} features",
        flush=True,
    )

    return X_treino_enc, X_val_enc, X_teste_enc


#validando com predeifnedSplit e com o gridSearch

def buscar_melhores_hiperparametros(
    X_treino_enc: np.ndarray,
    y_treino: np.ndarray,
    X_val_enc: np.ndarray,
    y_val: np.ndarray,
    param_grid: Dict[str, list],
) -> MLPClassifier:
  
    #concatena treino + validacao so num bloco
    X_treino_val: np.ndarray = np.vstack([X_treino_enc, X_val_enc])
    y_treino_val: np.ndarray = np.concatenate([y_treino, y_val])

    #vetor test_fold: -1 para treino, 0 para validacao
    test_fold = np.concatenate([
        np.full(len(X_treino_enc), -1, dtype=int),  #amostras de treino
        np.zeros(len(X_val_enc), dtype=int),         #amostras de validação
    ])

    ps = PredefinedSplit(test_fold)

    #early_stopping=True interrompe o treino quando a perda de validacaoo interna
    mlp_base = MLPClassifier(
        max_iter=2000,
        early_stopping=True,  #para automaticamente quando a melhora estagnar
        n_iter_no_change=20,  #paciencia: 20 epochs sem melhora
        random_state=42,
    )

    grid_search = GridSearchCV(
        estimator=mlp_base,
        param_grid=param_grid,
        cv=ps,        #usa o PredefinedSplit como estratégia de CV
        scoring="accuracy",
        n_jobs=1,     
        verbose=1,
    )

    print("\n[INFO] Iniciando GridSearchCV com PredefinedSplit...", flush=True)
    grid_search.fit(X_treino_val, y_treino_val)

    print(f"\n[RESULTADO] Melhor configuração encontrada:", flush=True)
    print(f"  {grid_search.best_params_}", flush=True)
    print(f"  Acurácia na validação: {grid_search.best_score_:.4f}", flush=True)

    return grid_search.best_estimator_


#avaliacao do modelo

def avaliar_modelo(
    modelo: MLPClassifier,
    X_teste_enc: np.ndarray,
    y_teste: np.ndarray,
    class_map: Dict[int, str],
    saida_matriz: str,
) -> None:
    
    #predições finais — só sobre o conjunto de teste
    y_pred_num: np.ndarray = modelo.predict(X_teste_enc)

    #converte id numerico pra descricao
    rotulos_ordenados: list[str] = [class_map[k] for k in sorted(class_map)]
    y_teste_str = np.array([class_map[v] for v in y_teste])
    y_pred_str  = np.array([class_map[v] for v in y_pred_num])

    #relatorio
    print("\n" + "=" * 60, flush=True)
    print("RELATÓRIO DE CLASSIFICAÇÃO — CONJUNTO DE TESTE", flush=True)
    print("=" * 60, flush=True)
    print(classification_report(y_teste_str, y_pred_str), flush=True)

    #matriz de confusao
    fig, ax = plt.subplots(figsize=(8, 6))

    ConfusionMatrixDisplay.from_predictions(
        y_teste_str,
        y_pred_str,
        display_labels=rotulos_ordenados,
        ax=ax,
        colorbar=True,
        xticks_rotation=30,
    )

    ax.set_title("Matriz de Confusão — MLP\n(Jogo da Velha)", fontsize=13, pad=14)
    plt.tight_layout()

    #salva na raiz do projeto
    plt.savefig(saida_matriz, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(
        f"\n[INFO] Matriz de Confusão salva em: {os.path.abspath(saida_matriz)}",
        flush=True,
    )

#pipeline principal
def main() -> None:

    X_treino, y_treino, X_val, y_val, X_teste, y_teste = carregar_dados(
        treino_path=TREINO_PATH,
        validacao_path=VALIDACAO_PATH,
        teste_path=TESTE_PATH,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
    )

    X_treino_enc, X_val_enc, X_teste_enc = preprocessar_features(
        X_treino, X_val, X_teste
    )

    #busca os hiperparametros
    melhor_modelo = buscar_melhores_hiperparametros(
        X_treino_enc=X_treino_enc,
        y_treino=y_treino,
        X_val_enc=X_val_enc,
        y_val=y_val,
        param_grid=PARAM_GRID,
    )

    #avaliacao final do modelo
    avaliar_modelo(
        modelo=melhor_modelo,
        X_teste_enc=X_teste_enc,
        y_teste=y_teste,
        class_map=CLASS_MAP,
        saida_matriz=SAIDA_MATRIZ,
    )


if __name__ == "__main__":
    main()

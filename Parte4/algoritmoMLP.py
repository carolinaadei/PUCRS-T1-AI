"""
algoritmoMLP.py
---------------
Classificação multiclasse do estado de um tabuleiro de Jogo da Velha
usando Multi-Layer Perceptron (MLPClassifier) do scikit-learn.

Fluxo geral:
    1. Carregamento dos dados (treino / validação / teste)
    2. Pré-processamento com OneHotEncoder (features categóricas)
    3. Busca de hiperparâmetros via GridSearchCV + PredefinedSplit
       (garante que o conjunto de validação físico seja usado como fold)
    4. Avaliação final exclusivamente no conjunto de teste
    5. Exportação da Matriz de Confusão como imagem PNG

Nota sobre labels:
    O target permanece NUMÉRICO durante todo o treinamento/busca, pois o
    `early_stopping` interno do MLPClassifier é incompatível com labels do
    tipo string (tentaria aplicar np.isnan em strings). A conversão para
    nomes descritivos ocorre APENAS na etapa de avaliação, usando CLASS_MAP.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")          # backend não-interativo: renderiza sem abrir janela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------------------------
# 0. CONSTANTES E CAMINHOS
# ---------------------------------------------------------------------------

# Raiz do projeto (pasta pai de Parte4/)
ROOT_DIR: str = os.path.join(os.path.dirname(__file__), "..")

TREINO_PATH: str = os.path.join(ROOT_DIR, "treino.csv")
VALIDACAO_PATH: str = os.path.join(ROOT_DIR, "validacao.csv")
TESTE_PATH: str = os.path.join(ROOT_DIR, "teste.csv")

# Arquivo de saída da Matriz de Confusão
SAIDA_MATRIZ: str = os.path.join(ROOT_DIR, "matriz_confusao_mlp.png")

# Colunas de features e target presentes nos CSVs
FEATURE_COLS: list[str] = ["tl", "tm", "tr", "ml", "mm", "mr", "bl", "bm", "br"]
TARGET_COL: str = "classe_id"

# Dicionário de mapeamento: inteiro → rótulo legível.
# Os IDs reais nos CSVs são 0–4; substitua os valores pelas regras do seu grupo.
CLASS_MAP: Dict[int, str] = {
    0: "Classe A",
    1: "Classe B",
    2: "Classe C",
    3: "Classe D",
    4: "Classe E",
}

# Grid de hiperparâmetros a explorar no GridSearchCV
PARAM_GRID: Dict[str, list] = {
    "hidden_layer_sizes": [(10,), (50,), (10, 10), (50, 25)],
    "activation": ["relu", "tanh"],
    "learning_rate_init": [0.001, 0.01],
}


# ---------------------------------------------------------------------------
# 1. CARREGAMENTO DOS DADOS
# ---------------------------------------------------------------------------

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
    """
    Lê os três arquivos CSV e separa features (X) do target (y).

    O target é mantido como array numérico inteiro para compatibilidade
    total com o MLPClassifier (incluindo early_stopping).
    A conversão para nomes descritivos é feita apenas na avaliação final.

    Parâmetros
    ----------
    treino_path, validacao_path, teste_path : str
        Caminhos para os respectivos arquivos CSV.
    feature_cols : list[str]
        Lista com os nomes das colunas de entrada.
    target_col : str
        Nome da coluna alvo.

    Retorna
    -------
    (X_treino, y_treino, X_val, y_val, X_teste, y_teste)
        X são DataFrames; y são arrays numpy de inteiros.
    """
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


# ---------------------------------------------------------------------------
# 2. PRÉ-PROCESSAMENTO — OneHotEncoder
# ---------------------------------------------------------------------------

def preprocessar_features(
    X_treino: pd.DataFrame,
    X_val: pd.DataFrame,
    X_teste: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aplica OneHotEncoder às features categóricas (valores 0, 1, 2).

    O encoder é **ajustado apenas no conjunto de treino** (fit) e depois
    aplicado (transform) em validação e teste, evitando vazamento de dados.

    `sparse_output=False` garante retorno de array denso (numpy).
    `handle_unknown='ignore'` evita erros caso alguma categoria inédita
    apareça nos conjuntos de validação ou teste.

    Parâmetros
    ----------
    X_treino, X_val, X_teste : pd.DataFrame
        Features brutas de cada partição.

    Retorna
    -------
    (X_treino_enc, X_val_enc, X_teste_enc) : np.ndarray
        Arrays codificados em one-hot.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    # Fit somente no treino para não vazar informações futuras
    X_treino_enc: np.ndarray = encoder.fit_transform(X_treino)
    X_val_enc: np.ndarray = encoder.transform(X_val)
    X_teste_enc: np.ndarray = encoder.transform(X_teste)

    print(
        f"[INFO] Dimensão após OneHotEncoder: {X_treino_enc.shape[1]} features",
        flush=True,
    )

    return X_treino_enc, X_val_enc, X_teste_enc


# ---------------------------------------------------------------------------
# 3. VALIDAÇÃO COM PredefinedSplit + GridSearchCV
# ---------------------------------------------------------------------------

def buscar_melhores_hiperparametros(
    X_treino_enc: np.ndarray,
    y_treino: np.ndarray,
    X_val_enc: np.ndarray,
    y_val: np.ndarray,
    param_grid: Dict[str, list],
) -> MLPClassifier:
    """
    Usa PredefinedSplit para forçar o GridSearchCV a usar exatamente o
    conjunto de validação físico como fold de avaliação, conforme exigido
    pelo enunciado do trabalho.

    Lógica do test_fold:
        -1  →  índice pertence ao conjunto de TREINO (nunca usado como fold)
         0  →  índice pertence ao conjunto de VALIDAÇÃO (fold único de avaliação)

    Parâmetros
    ----------
    X_treino_enc, X_val_enc : np.ndarray
        Features codificadas de treino e validação.
    y_treino, y_val : np.ndarray
        Targets numéricos correspondentes.
    param_grid : dict
        Grade de hiperparâmetros a explorar.

    Retorna
    -------
    best_model : MLPClassifier
        Modelo já treinado com os melhores hiperparâmetros encontrados.
    """
    # Concatena treino + validação em um único bloco
    X_treino_val: np.ndarray = np.vstack([X_treino_enc, X_val_enc])
    y_treino_val: np.ndarray = np.concatenate([y_treino, y_val])

    # Cria o vetor test_fold: -1 para treino, 0 para validação
    test_fold = np.concatenate([
        np.full(len(X_treino_enc), -1, dtype=int),  # amostras de treino
        np.zeros(len(X_val_enc), dtype=int),         # amostras de validação
    ])

    ps = PredefinedSplit(test_fold)

    # MLPClassifier base.
    # early_stopping=True interrompe o treino quando a perda de validação interna
    # para de melhorar — acelera muito o GridSearch. Só funciona com y numérico,
    # por isso o mapeamento para strings é feito apenas na avaliação final.
    mlp_base = MLPClassifier(
        max_iter=2000,
        early_stopping=True,  # para automaticamente quando a melhora estagna
        n_iter_no_change=20,  # paciência: 20 épocas sem melhora
        random_state=42,
    )

    grid_search = GridSearchCV(
        estimator=mlp_base,
        param_grid=param_grid,
        cv=ps,        # usa o PredefinedSplit como estratégia de CV
        scoring="accuracy",
        n_jobs=1,     # serial: evita overhead de multiprocessing no Windows
        verbose=1,
    )

    print("\n[INFO] Iniciando GridSearchCV com PredefinedSplit...", flush=True)
    grid_search.fit(X_treino_val, y_treino_val)

    print(f"\n[RESULTADO] Melhor configuração encontrada:", flush=True)
    print(f"  {grid_search.best_params_}", flush=True)
    print(f"  Acurácia na validação: {grid_search.best_score_:.4f}", flush=True)

    return grid_search.best_estimator_


# ---------------------------------------------------------------------------
# 4. AVALIAÇÃO NO CONJUNTO DE TESTE
# ---------------------------------------------------------------------------

def avaliar_modelo(
    modelo: MLPClassifier,
    X_teste_enc: np.ndarray,
    y_teste: np.ndarray,
    class_map: Dict[int, str],
    saida_matriz: str,
) -> None:
    """
    Realiza as predições finais no conjunto de teste e exibe / salva
    as métricas de avaliação.

    Aqui ocorre a única conversão numérico → string, tornando o relatório
    e a matriz de confusão completamente legíveis.

    Métricas geradas:
        - Relatório completo: Precision, Recall, F1-score, Accuracy
        - Matriz de Confusão visual salva como PNG

    Parâmetros
    ----------
    modelo : MLPClassifier
        Modelo treinado com os melhores hiperparâmetros.
    X_teste_enc : np.ndarray
        Features de teste já codificadas.
    y_teste : np.ndarray
        Labels numéricos verdadeiros do conjunto de teste.
    class_map : dict[int, str]
        Mapeamento de ID numérico para rótulo descritivo.
    saida_matriz : str
        Caminho completo onde a imagem PNG será salva.
    """
    # Predições finais — SOMENTE sobre o conjunto de teste
    y_pred_num: np.ndarray = modelo.predict(X_teste_enc)

    # Converte IDs numéricos para rótulos descritivos (apenas para exibição)
    rotulos_ordenados: list[str] = [class_map[k] for k in sorted(class_map)]
    y_teste_str = np.array([class_map[v] for v in y_teste])
    y_pred_str  = np.array([class_map[v] for v in y_pred_num])

    # --- Relatório de Classificação ---
    print("\n" + "=" * 60, flush=True)
    print("RELATÓRIO DE CLASSIFICAÇÃO — CONJUNTO DE TESTE", flush=True)
    print("=" * 60, flush=True)
    print(classification_report(y_teste_str, y_pred_str), flush=True)

    # --- Matriz de Confusão ---
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

    # Salva na raiz do projeto
    plt.savefig(saida_matriz, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(
        f"\n[INFO] Matriz de Confusão salva em: {os.path.abspath(saida_matriz)}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# 5. PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------

def main() -> None:
    """Ponto de entrada: orquestra todo o pipeline de ML."""

    # --- Carregamento ---
    X_treino, y_treino, X_val, y_val, X_teste, y_teste = carregar_dados(
        treino_path=TREINO_PATH,
        validacao_path=VALIDACAO_PATH,
        teste_path=TESTE_PATH,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
    )

    # --- Pré-processamento ---
    X_treino_enc, X_val_enc, X_teste_enc = preprocessar_features(
        X_treino, X_val, X_teste
    )

    # --- Busca de Hiperparâmetros ---
    melhor_modelo = buscar_melhores_hiperparametros(
        X_treino_enc=X_treino_enc,
        y_treino=y_treino,
        X_val_enc=X_val_enc,
        y_val=y_val,
        param_grid=PARAM_GRID,
    )

    # --- Avaliação Final ---
    avaliar_modelo(
        modelo=melhor_modelo,
        X_teste_enc=X_teste_enc,
        y_teste=y_teste,
        class_map=CLASS_MAP,
        saida_matriz=SAIDA_MATRIZ,
    )


if __name__ == "__main__":
    main()

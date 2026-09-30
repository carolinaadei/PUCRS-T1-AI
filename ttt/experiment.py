"""
Protocolo experimental comum a todos os algoritmos.

Antes cada script fazia a sua própria coisa: um usava GridSearchCV com `cv=5`,
outro com `cv=3`, outro escolhia o hiperparâmetro em um laço manual sobre a
validação. Isso viola o item 4 do enunciado, que pede os mesmos conjuntos e o
conjunto de teste reservado para a avaliação final.

Aqui o protocolo é único:

  1. treino + validação entram no GridSearchCV via `PredefinedSplit`, de modo
     que o conjunto de validação físico seja o único fold de pontuação;
  2. o melhor modelo é retreinado em treino + validação (`refit=True`);
  3. o conjunto de teste é tocado uma única vez, na avaliação final.
"""

import argparse

import joblib
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from ttt import dataset, evaluation, preprocessing
from ttt.config import ARTIFACTS_DIR


def treinar(nome, modelo, param_grid, abordagem):
    """Roda o protocolo completo para um algoritmo e devolve as métricas."""
    dados = dataset.carregar()
    X_tv, y_tv, test_fold = dataset.juntar_treino_validacao(dados)

    print(f"\n[INFO] {nome} — abordagem '{abordagem}': {preprocessing.ABORDAGENS[abordagem]}")
    print(f"[INFO] Treino: {len(dados.y_treino)} | "
          f"Validação: {len(dados.y_val)} | Teste: {len(dados.y_teste)}")

    # As chaves da grade ganham o prefixo do Pipeline aqui, para que cada script
    # declare os parâmetros pelo nome real do classificador.
    grid = {f"modelo__{k}": v for k, v in param_grid.items()}

    busca = GridSearchCV(
        preprocessing.criar_pipeline(abordagem, modelo),
        grid,
        cv=PredefinedSplit(test_fold),
        scoring="accuracy",
        refit=True,
        n_jobs=-1,
    )
    busca.fit(X_tv, y_tv)

    melhores = {k.removeprefix("modelo__"): v for k, v in busca.best_params_.items()}

    # Primeiro e único contato com o conjunto de teste
    y_pred = busca.best_estimator_.predict(dados.X_teste)
    resultado = evaluation.avaliar(
        nome, abordagem, dados.y_teste, y_pred, busca.best_score_, melhores
    )

    destino = ARTIFACTS_DIR / f"{nome.lower().replace(' ', '_')}_{abordagem}.joblib"
    joblib.dump(busca.best_estimator_, destino)
    print(f"[MODELO] artifacts/{destino.name}")

    return resultado


def main(nome, modelo, param_grid):
    """
    Ponto de entrada dos scripts em `models/`: lê a linha de comando e roda as
    abordagens pedidas.

    Rodar as duas é o padrão, porque é assim que se responde à pergunta do
    item 3 ("qual das abordagens é mais adequada").
    """
    parser = argparse.ArgumentParser(description=f"Treina e avalia o algoritmo {nome}.")
    parser.add_argument(
        "--abordagem",
        choices=[*preprocessing.ABORDAGENS, "ambas"],
        default="ambas",
        help="Abordagem de pré-processamento a utilizar (padrão: ambas)",
    )
    args = parser.parse_args()

    escolhidas = list(preprocessing.ABORDAGENS) if args.abordagem == "ambas" else [args.abordagem]
    resultados = [treinar(nome, modelo, param_grid, a) for a in escolhidas]

    if len(resultados) > 1:
        print(f"\n{'-' * 70}\nComparação das abordagens — {nome}\n{'-' * 70}")
        for r in resultados:
            print(f"  {r['abordagem']:<10} teste: {r['acuracia_teste']:.4f} | F1: {r['f1']:.4f}")
        melhor = max(resultados, key=lambda r: r["acuracia_teste"])
        print(f"\n  Melhor abordagem: '{melhor['abordagem']}' "
              f"(teste = {melhor['acuracia_teste']:.4f})")

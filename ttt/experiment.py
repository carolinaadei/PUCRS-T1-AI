"""
Protocolo experimental comum a todos os algoritmos.

Antes da reorganização cada script fazia a sua própria coisa: um usava
GridSearchCV com `cv=5`, outro com `cv=3`, outro escolhia o hiperparâmetro em
um laço manual sobre a validação. Isso viola o item 4 do enunciado, que pede os
mesmos conjuntos e o conjunto de teste reservado para a avaliação final.

Aqui o protocolo é único:

  1. treino + validação entram no GridSearchCV via `PredefinedSplit`, de modo
     que o conjunto de validação físico seja o único fold de pontuação;
  2. o melhor modelo é retreinado em treino + validação (`refit=True`);
  3. o conjunto de teste é tocado uma única vez, na avaliação final.

O pré-processamento entra dentro do Pipeline, então ele é reajustado a cada
fold e nunca enxerga os dados de validação antes da hora.
"""

import argparse

import joblib
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from ttt import dataset, evaluation, preprocessing
from ttt.config import ARTIFACTS_DIR, RANDOM_STATE, garantir_diretorios


def executar(
    nome: str,
    modelo: BaseEstimator,
    param_grid: dict,
    abordagem: str = "bruta",
    salvar_modelo: bool = True,
    n_jobs: int = -1,
) -> evaluation.Resultado:
    """
    Roda o protocolo completo para um algoritmo.

    Parâmetros
    ----------
    nome : str
        Nome do algoritmo, usado nos relatórios e nomes de arquivo.
    modelo : BaseEstimator
        Classificador ainda não treinado.
    param_grid : dict
        Grade de hiperparâmetros. As chaves são os nomes dos parâmetros do
        próprio classificador — o prefixo `modelo__` do Pipeline é aplicado
        aqui, para que cada script declare a grade sem se preocupar com isso.
    abordagem : str
        'bruta' ou 'derivada' (ver `ttt.preprocessing`).
    """
    dados = dataset.carregar()
    abord = preprocessing.obter(abordagem)

    print(f"\n[INFO] {nome} — abordagem '{abord.nome}': {abord.descricao}")
    print(f"[INFO] {dados.resumo()}")

    X_tv, y_tv, test_fold = dados.treino_mais_validacao()
    pipeline = abord.pipeline(modelo)

    # Traduz a grade para o namespace do Pipeline
    grid_pipeline = {f"modelo__{k}": v for k, v in param_grid.items()}
    n_combinacoes = _contar(grid_pipeline)
    print(f"[INFO] GridSearchCV: {n_combinacoes} combinações sobre o conjunto de validação...")

    busca = GridSearchCV(
        estimator=pipeline,
        param_grid=grid_pipeline,
        cv=PredefinedSplit(test_fold),
        scoring="accuracy",
        refit=True,
        n_jobs=n_jobs,
    )
    busca.fit(X_tv, y_tv)

    melhores = {k.removeprefix("modelo__"): v for k, v in busca.best_params_.items()}

    # Primeiro e único contato com o conjunto de teste
    y_pred = busca.best_estimator_.predict(dados.teste.X)

    resultado = evaluation.calcular(
        algoritmo=nome,
        abordagem=abord.nome,
        y_teste=dados.teste.y,
        y_pred=y_pred,
        acuracia_val=busca.best_score_,
        melhores_params=melhores,
    )
    evaluation.finalizar(resultado, dados.teste.y, y_pred)

    if salvar_modelo:
        garantir_diretorios()
        destino = ARTIFACTS_DIR / f"{nome.lower().replace(' ', '_')}_{abord.nome}.joblib"
        joblib.dump(busca.best_estimator_, destino)
        print(f"[MODELO] {destino.relative_to(ARTIFACTS_DIR.parent)}")

    return resultado


def cli(descricao: str) -> argparse.Namespace:
    """
    Argumentos comuns aos scripts de algoritmo.

    `--abordagem ambas` roda as duas representações em sequência, que é como se
    responde à pergunta do item 3 ("qual das abordagens é mais adequada").
    """
    parser = argparse.ArgumentParser(description=descricao)
    parser.add_argument(
        "--abordagem",
        choices=[*preprocessing.ABORDAGENS, "ambas"],
        default="ambas",
        help="Abordagem de pré-processamento a utilizar (padrão: ambas)",
    )
    parser.add_argument(
        "--sem-salvar",
        action="store_true",
        help="Não serializa o modelo treinado em artifacts/",
    )
    return parser.parse_args()


def executar_cli(nome: str, modelo_factory, param_grid: dict, args: argparse.Namespace) -> None:
    """
    Executa o algoritmo para as abordagens pedidas na linha de comando.

    `modelo_factory` é uma função sem argumentos que devolve um classificador
    novo — um por abordagem, para que não haja estado compartilhado entre as
    duas execuções.
    """
    escolhidas = list(preprocessing.ABORDAGENS) if args.abordagem == "ambas" else [args.abordagem]

    resultados = [
        executar(
            nome=nome,
            modelo=modelo_factory(),
            param_grid=param_grid,
            abordagem=abordagem,
            salvar_modelo=not args.sem_salvar,
        )
        for abordagem in escolhidas
    ]

    if len(resultados) > 1:
        print(f"\n{'-' * 70}\nComparação das abordagens — {nome}\n{'-' * 70}")
        for r in resultados:
            print(f"  {r.linha_resumo()}")
        melhor = max(resultados, key=lambda r: r.acuracia_teste)
        print(f"\n  Melhor abordagem: '{melhor.abordagem}' (teste = {melhor.acuracia_teste:.4f})")


def _contar(grid: dict) -> int:
    total = 1
    for valores in grid.values():
        total *= len(valores)
    return total


__all__ = ["executar", "executar_cli", "cli", "RANDOM_STATE"]

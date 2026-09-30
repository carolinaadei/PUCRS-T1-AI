# T1 — Tic Tac Toe com Machine Learning

PUCRS · Inteligência Artificial · Prof.ª Silvia Moraes

Sistema de IA que recebe o estado de um tabuleiro de jogo da velha 3×3 e o classifica
em uma de cinco categorias. A IA **não joga** — ela verifica o estado do jogo.

| id | Classe |
|---:|---|
| 0 | Empate |
| 1 | O vence |
| 2 | Possibilidade de Fim de Jogo |
| 3 | Tem jogo |
| 4 | X vence |

Codificação das casas do tabuleiro: `0 = vazio`, `1 = O`, `2 = X`.
Esse mapeamento é declarado **uma única vez**, em [ttt/config.py](ttt/config.py), e é a
fonte da verdade para todo o projeto — scripts, notebooks e front end.

---

## Como rodar

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

python models/comparar.py         # treina os 5 algoritmos e gera a comparação
python frontend/app.py            # sobe o front end em http://localhost:5001
```

Para rodar um algoritmo isolado:

```bash
python models/knn.py                       # ambas as abordagens de pré-processamento
python models/svm.py --abordagem derivada  # apenas uma delas
```

---

## Estrutura do projeto

A organização segue a ordem das etapas do enunciado: os dados entram em `data/`,
passam pelo pipeline de `ttt/`, são consumidos pelos algoritmos em `models/`, e os
resultados saem em `reports/` e no front end.

```
├── data/
│   ├── raw/              Dataset original do UCI, intocado
│   └── processed/        Divisão física: treino / validação / teste
│
├── ttt/                  Pacote compartilhado — o "como" do projeto
│   ├── config.py           Caminhos, CLASS_MAP, semente aleatória
│   ├── game_rules.py       Regras do jogo da velha (a verdade sobre um tabuleiro)
│   ├── dataset.py          Carga dos três conjuntos + PredefinedSplit
│   ├── preprocessing.py    As duas abordagens de pré-processamento
│   ├── experiment.py       Protocolo experimental único (GridSearch + avaliação)
│   └── evaluation.py       Métricas, matriz de confusão, registro dos resultados
│
├── models/               Um script por algoritmo — o "o quê" de cada um
│   ├── knn.py              k-NN               (exigido)
│   ├── arvore_decisao.py   Árvore de decisão  (exigido)
│   ├── mlp.py              MLP                (exigido)
│   ├── svm.py              SVM                (escolha livre)
│   ├── xgboost_clf.py      XGBoost            (escolha livre)
│   └── comparar.py         Roda todos e monta a comparação final
│
├── notebooks/
│   ├── 01_construcao_dataset.ipynb   Diagnóstico do UCI, reclassificação e divisão
│   └── 02_knn_exploratorio.ipynb     Exploração detalhada do k-NN
│
├── frontend/             Front end do item 6 (Flask + HTML)
├── reports/
│   ├── docs/             Documentação por algoritmo
│   ├── figures/          Matrizes de confusão e gráficos
│   └── metrics/          resultados.csv e comparacao_final.csv
└── artifacts/            Modelos treinados (.joblib), regeráveis
```

Cada script em `models/` declara apenas três coisas — `NOME`, `PARAM_GRID` e
`criar_modelo()`. Todo o resto (carga, pré-processamento, busca de hiperparâmetros,
avaliação e registro) vem de `ttt/`. É isso que garante que os cinco algoritmos sejam
medidos exatamente da mesma forma e que a comparação do item 5 signifique alguma coisa.

---

## Mapa: enunciado → código

| Item do enunciado | Onde está |
|---|---|
| 1. Objetivo — classificar o estado do tabuleiro | [ttt/game_rules.py](ttt/game_rules.py) define as 5 classes |
| 2. Dataset — obter, analisar e adequar o UCI | [notebooks/01_construcao_dataset.ipynb](notebooks/01_construcao_dataset.ipynb) |
| 3. Pré-processamento — duas abordagens | [ttt/preprocessing.py](ttt/preprocessing.py) |
| 4. Divisão do dataset — treino / validação / teste | [data/processed/](data/processed/) + [ttt/dataset.py](ttt/dataset.py) |
| 5. Solução de IA — 5 algoritmos e comparação | [models/](models/) + [reports/docs/](reports/docs/) |
| 6. Front end — interação e acurácia ao vivo | [frontend/app.py](frontend/app.py) |

---

## Dataset

O dataset do UCI ([tic+tac+toe+endgame](https://archive.ics.uci.edu/dataset/101/tic+tac+toe+endgame))
não atende ao problema como vem: ele traz **apenas 2 classes** (`positive`/`negative`) e
**somente estados finais** de partida. O notebook 01 documenta os problemas encontrados e
os passos executados para corrigi-los:

1. Reclassificação lógica dos 958 registros originais nas classes reais do trabalho;
2. Geração de estados sintéticos, por simulação de partidas, para as classes
   "Tem jogo" e "Possibilidade de Fim de Jogo", ausentes do UCI;
3. Deduplicação dos tabuleiros;
4. Balanceamento em ~200 amostras por classe;
5. Divisão estratificada em 60% treino / 20% validação / 20% teste.

**Limitação conhecida:** a classe *Empate* ficou com apenas **16 amostras** (contra 200
das demais), porque existem poucos tabuleiros de empate distintos. Isso torna as métricas
dessa classe pouco confiáveis — com ~4 amostras no teste, um único erro muda o recall em
25 pontos. As médias de precision/recall/F1 são `weighted` por causa disso.

---

## As duas abordagens de pré-processamento

O item 3 pede que se teste duas entradas diferentes e se verifique qual é mais adequada.
Ambas vivem em [ttt/preprocessing.py](ttt/preprocessing.py) e são intercambiáveis via
`--abordagem`:

| Abordagem | Entrada do modelo | Colunas |
|---|---|---|
| `bruta` | As 9 casas do tabuleiro em one-hot | 27 |
| `derivada` | Features extraídas: qtd. de X, qtd. de O, casas vazias, linhas com 2 X, linhas com 2 O, jogador da vez e máscara de posições ocupadas | 15 |

O one-hot na abordagem bruta evita que o modelo leia os códigos `0/1/2` como uma escala
ordenada — "X" não é o dobro de "O". Na abordagem derivada as features têm escalas
diferentes entre si (contagens de 0 a 9 ao lado de flags 0/1), então elas são
padronizadas.

`python models/comparar.py` imprime a acurácia média de cada abordagem ao final,
que é a resposta direta à pergunta do enunciado.

---

## Protocolo experimental

Vale para os cinco algoritmos, sem exceção ([ttt/experiment.py](ttt/experiment.py)):

1. Treino e validação entram no `GridSearchCV` via `PredefinedSplit`, de modo que o
   conjunto de validação **físico** seja o único fold de pontuação — e não uma
   reparticionamento aleatório do treino;
2. O melhor modelo é retreinado em treino + validação;
3. O conjunto de teste é tocado **uma única vez**, na avaliação final.

O pré-processamento entra dentro do `Pipeline`, então é reajustado a cada fold e nunca
enxerga os dados de validação antes da hora.

---

## Front end

O front end ([frontend/app.py](frontend/app.py)) põe um humano (X) contra um computador
que joga aleatoriamente (O). A cada jogada o tabuleiro é enviado ao classificador, e a
tela mostra a predição da IA, o estado real e a acurácia acumulada da sessão.

Conforme o enunciado, o fluxo é controlado pelo **estado real** e os erros da IA são
registrados: se a IA não detectar o fim de jogo, a partida encerra mesmo assim; se
detectar um fim que não ocorreu, o jogo continua.

O servidor **carrega** o melhor modelo de `artifacts/` em vez de treinar na subida.
Rode `python models/comparar.py` ao menos uma vez antes.

# T1 — Tic Tac Toe com Machine Learning

PUCRS · Inteligência Artificial · Prof.ª Silvia Moraes

**Grupo:** Carolina Gonçalves · Vicente Goldani

Sistema de IA que recebe o estado de um tabuleiro de jogo da velha 3×3 e o
classifica em uma de cinco categorias. **A IA não joga** — ela verifica o estado
do jogo.

| id | Classe | Definição |
|---:|---|---|
| 0 | Empate | tabuleiro cheio, sem vencedor |
| 1 | O vence | O tem três em linha |
| 2 | Possibilidade de Fim de Jogo | alguém pode vencer na próxima jogada |
| 3 | Tem jogo | partida em andamento, sem ameaça imediata |
| 4 | X vence | X tem três em linha |

Casas do tabuleiro: `0 = vazio`, `1 = O`, `2 = X`. Esse mapeamento é declarado
uma única vez, em [ttt/config.py](ttt/config.py), e é a fonte da verdade para
scripts, notebooks e front end.

---

## Como rodar

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

python models/comparar.py         # treina os 5 algoritmos e gera a comparação
python frontend/app.py            # front end em http://localhost:5001
```

Um algoritmo isolado:

```bash
python models/knn.py                       # ambas as abordagens
python models/svm.py --abordagem derivada  # apenas uma
```

---

## Documentação

O relatório técnico completo está em **[reports/docs/](reports/docs/)**, na ordem
das etapas do enunciado:

| Documento | Conteúdo |
|---|---|
| [1. Dataset](reports/docs/01-dataset.md) | o que o UCI entrega, os problemas encontrados, os passos de correção e por que "Empate" só tem 16 amostras |
| [2. Pré-processamento](reports/docs/02-preprocessamento.md) | as duas abordagens, o que cada uma aposta e como o custo é medido |
| [3. Metodologia](reports/docs/03-metodologia.md) | divisão dos dados, `PredefinedSplit`, métricas e cuidados contra overfitting |
| [4. Resultados](reports/docs/04-resultados.md) | comparação entre os algoritmos e escolha do melhor |
| [5. Front end](reports/docs/05-frontend.md) | a regra de teste do enunciado e os bugs corrigidos |
| [Algoritmos](reports/docs/README.md#algoritmos) | um documento por algoritmo, com parâmetros justificados e análise |

---

## Estrutura

Os dados entram em `data/`, passam pelo pipeline de `ttt/`, são consumidos pelos
algoritmos em `models/`, e os resultados saem em `reports/` e no front end.

```
├── data/
│   ├── raw/              Dataset original do UCI, intocado
│   └── processed/        Divisão física: treino / validação / teste
│
├── ttt/                  Código compartilhado — o "como"
│   ├── config.py           Caminhos, CLASS_MAP, semente aleatória
│   ├── game_rules.py       Regras do jogo: o estado real de um tabuleiro
│   ├── dataset.py          Carga dos três conjuntos + PredefinedSplit
│   ├── preprocessing.py    As duas abordagens de pré-processamento
│   ├── experiment.py       Protocolo experimental único
│   └── evaluation.py       Métricas, matriz de confusão, registro
│
├── models/               Um script por algoritmo — o "o quê"
│   ├── knn.py              k-NN               (exigido)
│   ├── arvore_decisao.py   Árvore de decisão  (exigido)
│   ├── mlp.py              MLP                (exigido)
│   ├── svm.py              SVM                (escolha livre)
│   ├── xgboost_clf.py      XGBoost            (escolha livre)
│   └── comparar.py         Roda todos e monta a comparação final
│
├── notebooks/
│   ├── 01_construcao_dataset.ipynb   Diagnóstico do UCI e divisão
│   └── 02_knn_exploratorio.ipynb     Exploração detalhada do k-NN
│
├── frontend/             Flask + HTML (item 6)
├── reports/
│   ├── docs/             Documentação técnica
│   ├── figures/          Matrizes de confusão e gráficos
│   └── metrics/          resultados.csv e comparacao_final.csv
└── artifacts/            Modelos treinados (.joblib), regeráveis
```

Cada script em `models/` declara três coisas — `NOME`, `MODELO` e `PARAM_GRID`.
Carga, pré-processamento, busca de hiperparâmetros, avaliação e registro vêm de
`ttt/`. É isso que garante que os cinco algoritmos sejam medidos exatamente da
mesma forma, e que a comparação do item 5 signifique alguma coisa.

---

## Mapa: enunciado → código

| Item do enunciado | Código | Documentação |
|---|---|---|
| 1. Classificar o estado do tabuleiro | [ttt/game_rules.py](ttt/game_rules.py) | — |
| 2. Dataset: obter, analisar, adequar | [notebooks/01_construcao_dataset.ipynb](notebooks/01_construcao_dataset.ipynb) | [01-dataset.md](reports/docs/01-dataset.md) |
| 3. Pré-processamento: duas abordagens | [ttt/preprocessing.py](ttt/preprocessing.py) | [02-preprocessamento.md](reports/docs/02-preprocessamento.md) |
| 4. Divisão: treino / validação / teste | [ttt/dataset.py](ttt/dataset.py), [data/processed/](data/processed/) | [03-metodologia.md](reports/docs/03-metodologia.md) |
| 5. Cinco algoritmos e comparação | [models/](models/) | [04-resultados.md](reports/docs/04-resultados.md) |
| 6. Front end com acurácia ao vivo | [frontend/app.py](frontend/app.py) | [05-frontend.md](reports/docs/05-frontend.md) |

---

## Resumo do processo

1. **O dataset do UCI não serve como vem.** Ele tem 2 classes e só posições
   finais de partida — zero instâncias de "Tem jogo" e "Possibilidade de Fim de
   Jogo". Reclassificamos as 958 linhas pelas regras do jogo, geramos estados
   intermediários por simulação de partidas, deduplicamos e balanceamos.
   → [01-dataset.md](reports/docs/01-dataset.md)

2. **"Empate" tem 16 amostras porque 16 é tudo que existe.** Enumerando os 126
   tabuleiros cheios possíveis, apenas 16 não têm vencedor. O dataset cobre
   100% dessa classe: o desbalanceamento é propriedade do problema, não falha do
   processo. Por isso as médias são `weighted`.

3. **Duas representações de entrada, um protocolo só.** A abordagem `bruta`
   entrega o tabuleiro em one-hot; a `derivada` entrega features extraídas
   (contagens, ameaças, jogador da vez). Tudo o mais é idêntico, para que a
   comparação fale da representação e não do resto.
   → [02-preprocessamento.md](reports/docs/02-preprocessamento.md)

4. **O conjunto de validação é físico, e é usado.** `PredefinedSplit` força o
   `GridSearchCV` a pontuar no `validacao.csv` em vez de reparticionar o treino.
   O teste é tocado uma única vez.
   → [03-metodologia.md](reports/docs/03-metodologia.md)

5. **A representação importou mais que o algoritmo.** Entregar as features
   prontas moveu a árvore de decisão de 54,9% para 86,0% — 31 pontos. Trocar de
   algoritmo, mantida a representação, move no máximo 11. A abordagem derivada
   venceu as duas perguntas do item 3: mais adequada e menos custosa.
   → [04-resultados.md](reports/docs/04-resultados.md)

6. **Toda feature esconde algo.** A abordagem derivada ganhou na média, mas
   nenhuma das suas 15 colunas codifica "três em linha" — e por isso o melhor
   modelo erra **todos** os empates. A bruta acerta os empates, mas confunde
   "Possibilidade" com "Tem jogo" 17 vezes.
   → [04-resultados.md §4.5](reports/docs/04-resultados.md)

---

## Resultado

**XGBoost com abordagem derivada: 89,0% de acurácia no conjunto de teste**
(F1 weighted 0,880), e 87,9% durante 100 partidas no front end. Tabela completa
dos cinco algoritmos nas duas abordagens em
[04-resultados.md](reports/docs/04-resultados.md).

---

## Ferramentas de IA utilizadas

> O enunciado pede que as ferramentas de IA usadas sejam indicadas, com os
> pontos em que foram aplicadas.

**Claude Code (Anthropic)** — usado em:

| Onde | O quê |
|---|---|
| Arquitetura | reorganização da estrutura de pastas e extração do código compartilhado em `ttt/` |
| `ttt/` | implementação dos módulos compartilhados: configuração, regras do jogo, carga, pré-processamento, protocolo e avaliação |
| `models/` | padronização dos cinco scripts sob uma interface comum |
| Revisão de código | identificação dos bugs descritos em [05-frontend.md](reports/docs/05-frontend.md) e do objetivo binário no XGBoost |
| Front end | redesign da interface e simplificação do JavaScript |
| Documentação | redação dos documentos em [reports/docs/](reports/docs/) |
| Análise do dataset | enumeração do espaço de estados do jogo ([01-dataset.md](reports/docs/01-dataset.md) §1.4) |
| Slides | montagem visual da apresentação |

O código original dos algoritmos (MLP, SVM, XGBoost) e os notebooks foram
escritos pelo grupo; a ferramenta foi usada para reorganizá-los, unificar o
protocolo experimental e documentar.

---

## Limitações conhecidas

| Limitação | Situação |
|---|---|
| "Empate" com 16 amostras | inerente ao problema; ver [01-dataset.md](reports/docs/01-dataset.md) §1.4 |
| A abordagem derivada erra todos os empates | falta uma feature de "três em linha"; ver [04-resultados.md §4.5](reports/docs/04-resultados.md) |
| `resultados.csv` da execução final não está commitado | rodar `python models/comparar.py` e versionar `reports/metrics/` e `reports/figures/` |
| Acurácia no front end não bate com a do teste | esperado, e explicado em [05-frontend.md](reports/docs/05-frontend.md) §5.5 |

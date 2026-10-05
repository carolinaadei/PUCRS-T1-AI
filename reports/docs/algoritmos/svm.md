# Support Vector Machine (SVM)

> Algoritmo de **escolha livre** do grupo. O enunciado pede que os dois
> algoritmos livres venham acompanhados de uma explicação de como funcionam.

**Código:** [models/svm.py](../../../models/svm.py)

```bash
python models/svm.py                      # roda as duas abordagens
python models/svm.py --abordagem bruta
```

---

## 1. Como funciona

A SVM procura o **hiperplano que separa as classes com a maior margem
possível** — isto é, a maior distância até os pontos mais próximos de cada lado.
Esses pontos mais próximos são os *vetores de suporte*, e são os únicos que
definem a fronteira; o resto do conjunto de treino não afeta o resultado.

Maximizar a margem é uma escolha de robustez: entre as infinitas fronteiras que
separam os dados de treino, a de maior margem é a que tende a errar menos em
dados novos.

**Quando os dados não são linearmente separáveis** — que é o caso aqui — a SVM
usa o *kernel trick*. Um kernel calcula o produto interno dos pontos como se
eles tivessem sido projetados num espaço de dimensão maior, onde a separação
linear é possível, **sem** calcular essa projeção explicitamente. É o que
permite fronteiras curvas a um custo tratável.

| Kernel | Fronteira que produz |
|---|---|
| `linear` | um hiperplano |
| `rbf` | fronteiras curvas locais, controladas por `gamma` |
| `poly` | fronteiras polinomiais |

**Vantagens**
- Forte em datasets pequenos e médios, que é exatamente o nosso caso (816
  amostras).
- A maximização da margem é uma regularização embutida, o que dá boa resistência
  a overfitting.
- Kernels dão flexibilidade sem mudar o algoritmo.

**Desvantagens**
- Muito sensível a `C` e `gamma`; sem busca de hiperparâmetros vai mal.
- Escala mal com o número de amostras (custo entre quadrático e cúbico).
- Não é interpretável, e não produz probabilidades calibradas por padrão.
- Foi concebida para 2 classes; problemas multiclasse são resolvidos por
  decomposição em vários classificadores binários (*one-vs-one*, no
  scikit-learn), o que multiplica o custo.

---

## 2. Pré-processamento

Rodado nas duas abordagens de [02-preprocessamento.md](../02-preprocessamento.md).

A SVM é, junto com o k-NN, o algoritmo **mais sensível à escala** do trabalho: o
kernel RBF calcula distâncias entre pontos, então uma feature com amplitude
maior dominaria a fronteira. As duas abordagens já entregam entradas em escala
adequada.

---

## 3. Parâmetros testados

```python
PARAM_GRID = {
    "kernel": ["linear", "rbf", "poly"],
    "C": [0.1, 1, 10, 50, 100],
    "gamma": ["scale", 0.01, 0.1, 1],
}
```

| Parâmetro | Valores | Por quê |
|---|---|---|
| `kernel` | `linear`, `rbf`, `poly` | o kernel entra na **grade**, e não é fixado à mão: assim a escolha é feita pelo conjunto de validação, com o mesmo critério dos demais parâmetros |
| `C` | 0,1 a 100 | controla o compromisso entre margem larga e erros no treino. `C` baixo aceita mais erros em troca de uma fronteira mais simples; `C` alto força acertar o treino e arrisca decorar. A faixa cobre duas ordens de grandeza para cada lado |
| `gamma` | `scale`, 0,01, 0,1, 1 | alcance da influência de cada amostra no kernel RBF. `gamma` alto produz fronteiras muito locais (risco de overfitting), `gamma` baixo produz fronteiras suaves. `scale` é a heurística do scikit-learn baseada na variância dos dados |

São 3 × 5 × 4 = **60 combinações**. Combinações com `kernel="linear"` ignoram
`gamma`, o que gera algumas repetições — desperdício pequeno, em troca de uma
grade que se lê de forma direta.

---

## 4. Resultados

| Abordagem | Acur. val. | Acur. teste | Precision | Recall | F1 | Treino (s) | Predição (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bruta` | 79,1% | **86,6%** | 0,867 | 0,866 | **0,866** | 0,0439 | 9,46 |
| `derivada` | 82,8% | 86,0% | 0,842 | 0,860 | 0,849 | 0,0142 | 3,99 |

**Melhores hiperparâmetros**

| Abordagem | Configuração |
|---|---|
| `bruta` | `kernel=rbf` · `C=10` · `gamma=scale` |
| `derivada` | `kernel=linear` · `C=0,1` |

### Análise

**O SVM é o único algoritmo que preferiu a abordagem bruta** (86,6% contra
86,0%) — e a diferença é de uma única amostra no teste, então as duas são
equivalentes na prática. O que interessa aqui não é qual venceu, mas **como** a
configuração mudou.

**A troca de kernel conta a história toda:**

| | `bruta` | `derivada` |
|---|---|---|
| Kernel | `rbf` (fronteiras curvas) | `linear` (um hiperplano) |
| `C` | 10 (aceita menos erros) | 0,1 (margem larga) |

Na abordagem bruta, o modelo precisa de fronteiras **curvas** e de um `C` alto
para separar as classes nas 27 colunas de one-hot. Na derivada, as features já
separam as classes tão bem que **um hiperplano simples basta** — e com `C=0,1`,
o menor valor da grade, indicando que uma margem larga foi suficiente, sem
precisar forçar o ajuste ao treino.

> **Isso é evidência direta de que as features derivadas fazem o trabalho que o
> kernel fazia.** O conhecimento do jogo, codificado em `linhas_2x` e
> `linhas_2o`, substitui a não linearidade que o RBF tinha que descobrir
> sozinho.

### Custo

A troca de kernel também derruba o custo:

| | `bruta` | `derivada` |
|---|---:|---:|
| Treino | 0,0439 s | **0,0142 s** (−68%) |
| Predição | 9,46 ms | **3,99 ms** (−58%) |

**O SVM na abordagem bruta tem a predição mais lenta de todas as dez
configurações (9,46 ms)**, o que é esperado: o kernel RBF exige calcular a
distância do ponto novo a cada vetor de suporte. Um kernel linear é apenas um
produto escalar.

### A classe difícil

O F1 de "Possibilidade de Fim de Jogo" subiu de **0,78 para 0,89** — o SVM já
era o melhor dos cinco nessa classe na abordagem bruta, e seguiu sendo o segundo
melhor na derivada, atrás do XGBoost (0,90).

### Correção de uma inconsistência na documentação anterior

A versão anterior deste documento afirmava que o encoding usado era
`x → 1, o → -1, b → 0`. Isso **não** correspondia ao código: o script aplicava
`{'x': 2, 'o': 1, 'b': 0}` através de um `.replace()` sobre dados que já estavam
numéricos — ou seja, a operação não fazia nada. O encoding efetivo era o dos
CSVs, seguido de `StandardScaler`.

Aquela execução, sob protocolo antigo (`cv=5` sobre o treino, ignorando o
conjunto de validação físico), registrou ~0,65 de acurácia. Sob o protocolo
unificado o SVM chega a 86,6% — **mais 21 pontos**, com o mesmo algoritmo. A
grade daquela execução antiga está em
[svm_gridsearch.csv](../../metrics/svm_gridsearch.csv), mantida só para
rastreabilidade.

> Comparação com os demais algoritmos: [04-resultados.md](../04-resultados.md).

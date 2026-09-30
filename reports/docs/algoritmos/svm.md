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

> ⚠️ **Números desatualizados — reexecutar.**
>
> As métricas abaixo foram obtidas **antes** da reorganização do projeto, sob um
> protocolo diferente: o script usava `cv=5` sobre o conjunto de treino e
> **ignorava o conjunto de validação físico**, o que contraria o item 4 do
> enunciado. O kernel também era escolhido à mão, fora da busca.
>
> Não são comparáveis com os demais algoritmos. Rode `python models/svm.py` e
> substitua esta seção pelos valores de
> [reports/metrics/resultados.csv](../../metrics/).

**Configuração apontada como melhor (protocolo antigo)**

| Parâmetro | Valor |
|---|---|
| `kernel` | `rbf` |
| `C` | 10 |
| `gamma` | `scale` |

| Métrica | Valor aproximado |
|---|---|
| Acurácia | ~0,65 |
| Precision | ~0,65 |

Grade completa daquela execução:
[reports/metrics/svm_gridsearch.csv](../../metrics/svm_gridsearch.csv) —
a melhor pontuação registrada ali foi 0,6524, com `C=10` e `gamma=scale`.

Figuras daquela execução:
[svm_matriz_confusao.png](../../figures/svm_matriz_confusao.png) ·
[svm_comparacao_modelos.png](../../figures/svm_comparacao_modelos.png)

### Correção de uma inconsistência na documentação anterior

A versão anterior deste documento afirmava que o encoding usado era
`x → 1, o → -1, b → 0`. Isso **não** correspondia ao código: o script aplicava
`{'x': 2, 'o': 1, 'b': 0}` através de um `.replace()` sobre dados que já estavam
numéricos — ou seja, a operação não fazia nada. O encoding efetivo era o dos
CSVs (`0` vazio, `1` O, `2` X), seguido de `StandardScaler`.

### Tabela a preencher

| Abordagem | Acurácia val. | Acurácia teste | Precision | Recall | F1 | Treino (s) |
|---|---:|---:|---:|---:|---:|---:|
| `bruta` | | | | | | |
| `derivada` | | | | | | |

### O que verificar ao rodar

1. **Qual kernel a validação escolhe agora**, com o one-hot em vez do
   `StandardScaler` sobre códigos brutos. É possível que o `linear` se torne
   competitivo, já que o one-hot expande o espaço de 9 para 27 dimensões e
   separações lineares ficam mais fáceis em dimensão alta.
2. **O custo de treino entre as abordagens.** A SVM é o algoritmo em que a
   diferença deve ser mais visível, porque seu custo cresce com o número de
   features: 27 colunas (bruta) contra 15 (derivada).
3. **A classe Empate.** Com 9 amostras de treino contra 480 das outras, e sem
   `class_weight="balanced"`, é provável que a SVM raramente a prediga.

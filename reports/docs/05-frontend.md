# 5. Front end

> Item 6 do enunciado — front end mínimo onde um humano joga contra a máquina; a
> IA indica o estado do jogo a cada jogada, o front end dá feedback ao usuário, e
> a acurácia da IA durante a interação é contabilizada.

**Servidor:** [frontend/app.py](../../frontend/app.py) ·
**Interface:** [frontend/templates/index.html](../../frontend/templates/index.html)

```bash
python models/comparar.py     # treina os modelos (uma vez)
python frontend/app.py        # http://localhost:5001
```

---

## 5.1 O que a tela mostra

| Elemento | Conteúdo |
|---|---|
| Tabuleiro | 3×3 clicável; X é o humano, O é o computador |
| Predição da IA | a classe que o modelo respondeu para o tabuleiro atual |
| Veredito | acerto ou erro, comparando a predição com o estado real |
| Estado real | a classe correta, calculada pelas regras do jogo |
| Acurácia da sessão | percentual acumulado de acertos da IA, com barra |
| Histórico | um ponto por predição, verde para acerto e vermelho para erro |
| Modelo em uso | qual artefato está servindo as predições |

O humano joga clicando numa casa vazia. O computador joga ao clique do botão, e
escolhe uma casa vazia **aleatoriamente**, como o enunciado pede.

---

## 5.2 A regra de teste do enunciado

O enunciado define um comportamento específico, e é o ponto mais importante
desta parte:

> Ao testar a IA no Front end, encerre o jogo quando ela **não** detectar o fim
> de jogo. E continue o jogo se a IA detectar o fim de jogo **incorretamente**.

Ou seja: **quem controla o fluxo é o estado real, não a predição da IA.** A
predição é observada e pontuada, mas não decide nada. Isso é o que permite medir
a acurácia da IA sem que os erros dela travem a partida.

Implementado em `aposJogada()`, no
[template](../../frontend/templates/index.html):

```javascript
const resposta = await chamar('/api/classify');
mostrarPredicao(resposta);          // registra acerto ou erro

if (resposta.fim_de_jogo) {         // <- estado REAL, não resposta.class_id
  emJogo = false;
  mostrarResultado(resposta.real_id, !resposta.correto);
  return;
}
```

Os dois casos se resolvem sozinhos:

| Situação | O que acontece |
|---|---|
| A IA **não** detecta o fim, mas o jogo acabou | `fim_de_jogo` é `true`, a partida encerra, e a tela avisa que a IA errou ao classificar o fim |
| A IA detecta fim que **não** houve | `fim_de_jogo` é `false`, o jogo continua, e o erro é contabilizado |

---

## 5.3 Arquitetura

### O servidor carrega o modelo, não treina

A função `carregar_melhor_modelo()` em [frontend/app.py](../../frontend/app.py)
lê [reports/metrics/resultados.csv](../metrics/), ordena por acurácia de teste e
carrega o primeiro artefato disponível em [artifacts/](../../artifacts/).

Isso resolve dois problemas de uma versão anterior, que treinava um MLP a cada
inicialização: a subida era lenta, e o front end acabava usando um modelo
diferente do que foi avaliado no relatório. Agora o modelo servido é
**exatamente** o vencedor da comparação do item 5.

### O cliente não conhece as regras do jogo

`/api/classify` devolve a predição **e** o estado real:

```json
{
  "class_id": 3,
  "class_name": "Tem jogo",
  "real_id": 2,
  "real_name": "Possibilidade de Fim de Jogo",
  "correto": false,
  "fim_de_jogo": false
}
```

O estado real é calculado no servidor, por `classificar_id` em
[ttt/game_rules.py](../../ttt/game_rules.py) — as mesmas regras que rotularam o
dataset. O JavaScript não reimplementa nada disso; ele só calcula qual linha
destacar visualmente quando alguém vence.

Isso importa porque **uma regra duplicada é uma regra que vai divergir**. A
versão anterior mantinha `checkWinner`, `canWinNext` e uma tabela de rótulos no
cliente, e foi justamente ali que apareceram os dois bugs descritos abaixo.

---

## 5.4 Bugs corrigidos nesta parte

Vale registrar no relatório, porque são erros silenciosos — a tela parecia
funcionar.

### Bug 1 — X e O trocados na entrada do modelo

O cliente codificava o humano (X) como `1` e o computador (O) como `2`. O
dataset usa o **oposto**: `1` é O e `2` é X. O modelo recebia todos os
tabuleiros com os jogadores invertidos.

O efeito era sutil porque um segundo erro o mascarava: a tabela de rótulos do
servidor também estava invertida (`1` rotulado como "X Vence" e `4` como
"O Vence"). Nas telas de vitória os dois erros se cancelavam e o resultado
parecia certo.

O que **não** se cancelava era a paridade de turnos. O modelo aprendeu que X
abre a partida, logo `qtd_x == qtd_o` ou `qtd_x == qtd_o + 1`. Com os jogadores
trocados, o modelo recebia tabuleiros que violam essa regra e que ele nunca viu
no treino — degradando justamente "Tem jogo" e "Possibilidade de Fim de Jogo".

**Correção:** a codificação do cliente passou a ser a mesma do treino, e os
rótulos deixaram de existir no cliente — vêm da resposta da API.

### Bug 2 — mapeamento de classes duplicado e divergente

O mesmo `CLASS_MAP` existia em três lugares com valores diferentes. Hoje ele é
declarado uma única vez, em [ttt/config.py](../../ttt/config.py), e foi validado
contra os dados: as regras do jogo reproduzem 816 de 816 rótulos dos CSVs.

---

## 5.5 Acurácia durante a interação

> O enunciado pede que os acertos e erros durante a interação sejam
> contabilizados e registrados no relatório.

A tela contabiliza ao vivo e acumula ao longo da sessão, atravessando várias
partidas. O contador não zera ao reiniciar a partida; só o histórico visual.

### Resultado medido

100 partidas completas pela API do front end (`/api/classify`), com jogadas
aleatórias dos dois lados, servindo o modelo **XGBoost · derivada**:

| | |
|---|---|
| Acurácia na interação | **87,9%** |
| Predições corretas | 667 de 759 |
| Acurácia no conjunto de teste | 89,0% |

**Acerto por estado real**

| Estado real | Predições | Acerto |
|---|---:|---:|
| O vence | 35 | 100% |
| Tem jogo | 285 | 98% |
| X vence | 56 | 96% |
| Possibilidade de Fim de Jogo | 374 | 80% |
| **Empate** | **9** | **0%** |

### Análise

**A acurácia na interação (87,9%) ficou perto da de teste (89,0%), mas pelas
razões erradas** — e é isso que torna essa medição interessante para o
relatório.

As duas distribuições são **muito diferentes**:

| Classe | No conjunto de teste | Numa partida real |
|---|---:|---:|
| Possibilidade de Fim de Jogo | 24,4% | **49,3%** |
| Tem jogo | 24,4% | 37,5% |
| Empate | 2,4% | 1,2% |

O conjunto de teste é balanceado por construção (40 amostras por classe, exceto
Empate). Uma partida real passa metade do tempo em "Possibilidade de Fim de
Jogo", visita "Tem jogo" no começo, e toca um estado terminal **uma única vez**,
no fim.

Como "Possibilidade" é a classe mais difícil (80% de acerto) **e** a mais
frequente numa partida, era de esperar que a acurácia na interação caísse bem
abaixo da de teste. Ela não caiu porque "Tem jogo" (98%) é a segunda mais
frequente e compensa.

**Os 0% em Empate confirmam o ponto cego da abordagem derivada.** Das 759
predições, 9 eram empates e o modelo errou todas — exatamente o comportamento
descrito em [04-resultados.md §4.5](04-resultados.md): nenhuma das 15 features
derivadas codifica "três em linha", então um tabuleiro cheio sem vencedor fica
indistinguível de uma vitória de X.

> **Conclusão para o relatório:** a métrica de teste, sozinha, não descreve o
> desempenho em uso. Um modelo com 89% no teste erra 100% dos empates em
> partidas reais — e só não parece pior porque empates são raros. Se o critério
> de escolha tivesse incluído a distribuição real de uso, a abordagem `bruta`
> (que acerta os 4 empates do teste) seria a escolha mais defensável.

## 5.6 Endpoints

| Rota | Método | Entrada | Saída |
|---|---|---|---|
| `/` | GET | — | a página |
| `/api/classify` | POST | `{"board": [9 inteiros]}` | predição, estado real, se acertou, se acabou |
| `/api/computer_move` | POST | `{"board": [9 inteiros]}` | `{"move": índice}` ou `-1` se não há casa vazia |

A entrada é validada em `_ler_tabuleiro`: exige uma lista de 9 posições com
valores `0`, `1` ou `2`, e devolve HTTP 400 com mensagem caso contrário.

---

## 5.7 Design

Interface limpa e minimalista, com cor nos pontos de interação: X em indigo, O
em âmbar, veredito em verde ou vermelho, animação na casa recém-marcada e
destaque na linha vencedora. O layout é responsivo e funciona em largura de
celular.

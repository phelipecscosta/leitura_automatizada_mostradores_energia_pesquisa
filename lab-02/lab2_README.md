# Lab02 — Triagem profunda

Entrega do Marco 1 (Lab02) da disciplina de Deep Learning, no projeto de
leitura automatizada de fotografias de medidores de uma concessionária de
energia do Nordeste. As fotos não fazem parte deste repositório; os
rótulos usam códigos pseudônimos (`rotulos/lab2_esquema.md`).

## 1. Mudanças em relação às arquiteturas originais

As arquiteturas originalmente apresentadas, A (divisão de classes,
"Arquitetura I") e B (multiclasse, "Arquitetura II"), foram revisadas.
**A arquitetura adotada e construída primeiro é a B aperfeiçoada.** A A
melhorada será construída depois, se for oportuno e viável, e comparada com
a B no mesmo lote de teste e na mesma métrica. As principais melhorias foram
a verificação da leitura guiada pelo CSV e a retirada do YOLO.

### 1.1 Resposta às críticas recebidas

| Crítica | O que mudou |
| --- | --- |
| O banco de imagens rotuladas não se confirmou | Rotulagem própria de ao menos 300 fotos, estratificadas pelos 4 lotes, com 50 rotuladas às cegas por duas pessoas e kappa de Cohen por classe. A partição é por lote. O CSV passa a ser usado como rótulo fraco e como entrada de verificação |
| Muitos estágios sem conta de propagação de erro | Cada estação tem o destino do erro definido. As rejeições antecipadas só removem casos de alta confiança, e os duvidosos seguem para a leitura ou para "Verificar". O laço "Pode melhorar?" foi removido, por não ter critério de parada. A conta do produto das taxas está na seção 3 da ficha |
| O YOLO exige caixas e tem questão de licença | O YOLO (Ultralytics, AGPL-3.0) foi substituído por detectores do `torchvision` (BSD-3): SSDlite320 com MobileNetV3-Large no perfil CPU e Faster R-CNN com MobileNetV3-Large-FPN como candidato ao perfil GPU. As caixas continuam necessárias; o custo de anotá-las é medido no item N1 |
| As variantes não foram comparadas | A B é construída primeiro e a A melhorada serve de comparação. A pergunta experimental é: especializar detector e leitor por tipo de medidor compensa o custo de dobrar os modelos? |

### 1.2 Por que a B antes da A

| Critério | B (multiclasse) | A (divisão por tipo) |
| --- | --- | --- |
| Modelos aprendidos | 3: triagem, detector, leitor | 5: triagem, 2 detectores, 2 leitores |
| Uso dos dados rotulados | Detector e leitor treinados com todas as fotos | Caixas e transcrições divididas entre os dois tipos, antes de se conhecer a proporção de ciclométricos |
| Ponto forte | Simplicidade; detecta divergência entre foto e registro | Leitor especializado para 7 segmentos e para tambores |
| Risco principal | Viés de confirmação; leitor único para dois tipos de caractere | Mais fatores de erro em série; poucos exemplos se os ciclométricos forem raros |

A ordem não desperdiça trabalho. As caixas anotadas para a B (display
digital, display ciclométrico e placa) e os recortes transcritos servem para
treinar os detectores e leitores da A, bastando separá-los por tipo.
Construir a A depois custa treino e integração, não uma nova rotulagem.

A A será construída se a validação da B mostrar uma diferença de acerto de
leitura entre medidores digitais e ciclométricos acima de um limite definido
no protocolo de avaliação. Essa diferença é o sintoma que a especialização
da A resolveria.

## 2. Protocolo de avaliação da triagem

Fixado em 08/10/2026, antes de qualquer treino. Mudanças posteriores entram
como revisão datada, com o motivo, sem apagar o texto original.

### 2.1 Dados

- Avaliação: as 225 fotos aleatórias rotuladas dos lotes 20/05, 21/05 e
  22/05. Rótulo final: o da adjudicação (ADJ), se existir; senão, o do R1.
- Lote de teste (03/07): não é lido nesta etapa. É aberto uma única vez,
  na ET4, para as triagens de DL e de ML.
- Zeros sem nota (36 fora do teste): entram no treino quando o lote deles
  está no treino; são preditos fora da dobra e reportados à parte (V4);
  ficam fora das métricas principais. Os 14 do lote de teste seguem lacrados.
- Sem enriquecimento de "outros". A especificidade é reportada como não
  estimável (18 fotos).

### 2.2 Validação

- Três dobras, uma por lote: cada modelo treina em dois lotes e prediz o
  terceiro. As predições fora da dobra das 225 fotos são reunidas.
- Três sementes por configuração (0, 1 e 2). Métricas reportadas como
  média e desvio entre sementes.

### 2.3 Modelos comparados

- Profundo (extração de características): espinha dorsal pré-treinada no
  ImageNet, congelada; características calculadas uma vez por foto;
  cabeças lineares.
- Baseline não profunda: HOG, histograma de cor HSV e estatísticas de
  qualidade (brilho, contraste e variância do Laplaciano); cabeças lineares.
- Os dois recebem a foto com o realce fixo (autocontraste, cutoff=1).
- Resolução do profundo: nativa e 224 px, sem distorcer a proporção.

### 2.4 Cabeças e perdas (iguais nos dois modelos)

- Cena: 3 saídas (digital, ciclométrico, outros), entropia cruzada sobre
  logits. Indeterminado: perda parcial −log(p_digital + p_ciclométrico).
  Peso por classe proporcional a 1/frequência no treino de cada dobra.
- Legibilidade: 1 saída, `BCEWithLogitsLoss`, mascarada em "outros";
  `pos_weight` = negativos/positivos no treino de cada dobra.
- Perda total: cena + legibilidade (λ = 1).

### 2.5 Hiperparâmetros

- Fixos, sem busca: AdamW, taxa 1e-3, decaimento 1e-4, lote completo,
  300 épocas, sem parada antecipada. Com a validação por lote, escolher
  hiperparâmetros ou a época de parada pela dobra de avaliação
  contaminaria as predições fora da dobra.

### 2.6 Métricas

- Otimizadora: média da precisão média (AP) de "outros" e da AP de
  "ilegível", fora da dobra.
- Secundárias (diagnóstico): perda logarítmica de cada cabeça; AP por lote.
- Intervalos de 95% por bootstrap sobre as fotos (2.000 reamostragens).

### 2.7 Regra de vitória

- Diferença (profundo − baseline) por bootstrap pareado, nas mesmas fotos,
  com a probabilidade média das três sementes.
- Vence o profundo se o limite inferior for maior que zero; vence a
  baseline se o limite superior for menor que zero; caso contrário, empate.
- Empate: segue o profundo.
- Resolução (nativa × 224 px): mesma regra; em empate, segue 224 px, pelo
  menor custo em CPU.

### 2.8 Suficiência de rótulos

- Curva da métrica otimizadora com 25%, 50%, 75% e 100% do treino de cada
  dobra. Se a métrica subir entre 75% e 100% mais que a meia-largura do
  seu intervalo, os rótulos são registrados como insuficientes.

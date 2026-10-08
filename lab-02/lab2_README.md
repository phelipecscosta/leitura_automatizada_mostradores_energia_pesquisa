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

### 2.9 Revisões do protocolo

| Data | Revisão | Motivo |
| --- | --- | --- |
| 08/10/2026 | Na seção 2.1, a especificidade se apoia em 17 fotos de "outros" da amostra aleatória fora do teste, e não em 18. A 18ª está no conjunto de estresse (22/05) e fica fora das métricas principais | Contagem conferida pelo carregador de rótulos (T3.1), antes de qualquer treino. Nenhuma regra muda |
| 08/10/2026 | Seção 2.3: a espinha dorsal deixa de ser única. MobileNetV3-Large e EfficientNet-B0 (pesos ImageNet do `torchvision`) são comparadas. Seção 2.7: ordem fixa das comparações: (1) espinha dorsal, as duas a 224 px, empate segue a MobileNetV3-Large; (2) resolução, nativa × 224 px, só com a espinha vencedora, empate segue 224 px; (3) a configuração profunda vencedora × baseline, empate segue o profundo. Todas com a regra de vitória da seção 2.7 | Escolha da espinha dorsal com evidência medida, e não só por custo. Revisão feita antes de qualquer treino. O desempate pela MobileNetV3-Large se apoia no menor custo no piso da E04 e na mesma família do detector do perfil CPU (E07) |
| 08/10/2026 | Seção 2.3, baseline: calculada sobre a mesma imagem que o modelo profundo recebe a 224 px (realce fixo, 299 × 224). HOG em tons de cinza, 9 orientações, células de 32 × 32 px, blocos de 2 × 2 células, normalização L2-Hys (1.728 dimensões); histograma conjunto HSV com 8 × 4 × 4 faixas, normalizado (128); brilho médio, contraste e variância do Laplaciano em tons de cinza (3). Total: 1.859 dimensões. Seções 2.3 e 2.4: nos dois modelos, as características são padronizadas por escore z com a média e o desvio do treino de cada dobra. Seção 2.4: o indeterminado recebe o peso médio de um medidor de tipo conhecido (média dos pesos de digital e ciclométrico, ponderada pelas contagens) | Lacunas do protocolo preenchidas antes de qualquer execução do D3. Células de 32 px espelham o campo de cada posição do último mapa das CNNs (B1) e mantêm a dimensão na ordem das características profundas. A padronização só com o treino evita vazamento da dobra de avaliação |
| 08/10/2026 | Nova seção 2.10: saída de decisão da triagem (escore único de rejeição), métricas da curva cobertura × risco, metas Y = 10% e R = 0,97 verificadas pelo limite superior do intervalo, e reformulação da R3 da E09 | Itens E1 e E2 da rubrica. A R3 mede leituras aprovadas, que a triagem não produz (E23). Fixado antes de qualquer calibração ou curva |
| 08/10/2026 | Nova seção 2.11 (calibração). Seções 2.7 e 2.10: a partir da calibração, as três sementes são combinadas pela média dos logits, e não pela média das probabilidades. As comparações do D3, já feitas com a média das probabilidades, não são refeitas | Com cabeças lineares sobre as mesmas características, a média dos logits equivale a uma cabeça linear com a média dos pesos: o modelo avaliado na E1 e na E2 é o mesmo que é exportado e entregue. Fixado antes de qualquer calibração |
| 08/10/2026 | Seção 2.11, revisão feita **depois** do primeiro resultado da calibração: antes da temperatura, os logits recebem a correção dos pesos de classe usados no treino de cada dobra (logit − log w_c na cena; logit − log pos_weight na legibilidade). O peso do indeterminado não entra, porque não corresponde a um logit. A temperatura segue como fixado. O notebook reporta três versões: antes, só temperatura, e correção mais temperatura | Na validação, a média prevista de P(outros) era 2,2 vezes a prevalência antes da temperatura e 3 vezes depois, e o ECE de P(rejeitável) piorou de 0,096 para 0,163. Causa: a entropia cruzada ponderada aprende probabilidades proporcionais a w_c × p(c \| foto) (Elkan, 2001; Menon et al., 2021). A correção não ajusta nenhum parâmetro novo, e o teste continua lacrado |


### 2.10 Confiança e recusa (E1 e E2)

Fixado em 08/10/2026, antes de qualquer calibração ou curva cobertura × risco.

**Saída de decisão.** A triagem só decide rejeitar. Seguir adiante é o
caminho padrão e seguro, porque a foto ainda passa pelo detector, pelo
leitor ou pela verificação humana.

- Escore de rejeição: P(rejeitável) = p_outros + (1 − p_outros) × p_ilegível,
  com as probabilidades calibradas (E1). Como p_ilegível é condicional a
  ser medidor, o escore é a probabilidade total de dois eventos disjuntos:
  não ser medidor, ou ser medidor ilegível.
- Regra: rejeita se P(rejeitável) ≥ t. Destino: "Rejeitadas / Outros" se
  p_outros ≥ (1 − p_outros) × p_ilegível; senão, "Rejeitadas / Baixa
  qualidade".
- Probabilidades: média das três sementes, como na seção 2.7.

**Métricas** (amostra aleatória, predições fora da dobra; os zeros ficam
fora, conforme a seção 2.1):

| Métrica | Definição | Papel |
| --- | --- | --- |
| Rejeitável | Foto de "outros" ou medidor ilegível | Universo da cobertura |
| Cobertura da rejeição | Rejeitáveis rejeitadas ÷ rejeitáveis | Maximizar |
| Risco | Medidores legíveis rejeitados ÷ fotos rejeitadas | ≤ Y |
| Perda | Medidores legíveis rejeitados ÷ medidores legíveis | ≤ 1 − R |
| Destino trocado | Medidores ilegíveis enviados a "Outros" ÷ medidores ilegíveis rejeitados | Só reportado |

- Só a rejeição de um medidor legível é erro, porque descarta informação.
  Um medidor ilegível enviado a "Outros" não descarta informação e fica
  fora do risco: o princípio 1 da E09 é lido pela perda de informação, e
  não pela pasta de destino.
- Uma foto de "outros" que segue adiante não é erro nesta curva
  (tolerância da E09).
- A cobertura da rejeição não é a cobertura da E09 (aprovadas ÷ legíveis),
  que só existe com o leitor e os validadores.

**Metas.** Y = 10% e R = 0,97; a precisão da rejeição (P da E09) é
1 − Y = 90%.

**Verificação.** Risco e perda pelo limite superior unilateral de 95%
(Clopper-Pearson). Entre os limiares t que cumprem as duas metas, vale o
de maior cobertura da rejeição. Se nenhum cumprir, a meta é declarada fora
da curva: o README informa a maior cobertura possível com cada restrição
isolada e quantas rejeições sem erro tornariam a meta demonstrável.

**Teste (ET4).** O limiar é aplicado sem ajuste, e cobertura, risco e
perda são reportados com intervalo. O tamanho do teste limita o risco que
ele consegue demonstrar; o limite é declarado com a contagem exata de
rejeitáveis.

**Reformulação da R3 (E09 e E23).** A R3 mede leituras aprovadas e só pode
ser avaliada com o leitor e os validadores (ET7). No Lab02, a restrição de
risco da curva cobertura × risco é a definida acima.

**Se a meta não couber.** O resultado é registrado como insuficiência de
rótulos para a decisão automática, com a conta acima. A ampliação da
rotulagem fica para depois, com Y e R mantidos e o lote de teste intocado.

### 2.11 Calibração (E1)

Fixado em 08/10/2026, antes de qualquer calibração.

**Entrada.** Logits fora da dobra das três sementes, combinados pela média
(seção 2.9, revisão de 08/10), só da amostra aleatória (os zeros ficam
fora, conforme a seção 2.1).

**Temperatura.** Uma por cabeça, T > 0, aplicada como logit ÷ T:

- Cena: minimiza a perda logarítmica da softmax, com a regra parcial do
  indeterminado (−log(p_digital + p_ciclométrico)).
- Legibilidade: minimiza a perda logarítmica da sigmoide, só nos medidores.
- Sem pesos de classe: a calibração deve refletir as frequências reais.

**Validação aninhada por lote.** Para cada lote, as temperaturas são
ajustadas nos outros dois lotes e aplicadas ao lote deixado de fora. O ECE
"depois" e todas as métricas da E2 (seção 2.10) usam essas predições. As
temperaturas de implantação são ajustadas com os três lotes e reportadas
ao lado das três intermediárias.

**ECE.** Binário, 10 faixas de largura igual, média das diferenças
ponderada pelo número de fotos em cada faixa, com intervalo de 95% por
bootstrap (2.000 reamostragens).

| Quantidade | Rótulo | Fotos | Papel |
| --- | --- | --- | --- |
| P(rejeitável) (seção 2.10) | Rejeitável | Todas | Principal |
| P(outros) | Outros | Todas | Secundária |
| P(ilegível \| medidor) | Ilegível | Medidores | Secundária |

Para cada quantidade: ECE e diagrama de confiabilidade antes e depois da
temperatura.

**Limitação declarada.** As temperaturas vêm de cabeças treinadas com dois
lotes e são aplicadas ao modelo final, treinado com três (E25).

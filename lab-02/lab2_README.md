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

## 3. Itens N e V adaptados

A arquitetura mudou em relação às propostas originais (seção 1). Os itens N0
a N4 e V0 a V4 da rubrica foram combinados e adaptados à arquitetura B
aperfeiçoada, conforme o estado real do projeto: a triagem está treinada e
avaliada; o detector e o leitor serão construídos nas etapas seguintes.

### 3.1 N0 e V0 — Bloco profundo: triagem por transferência

O primeiro bloco profundo é a triagem "medidor (digital, ciclométrico) /
outros" com a cabeça "legível / ilegível": MobileNetV3-Large pré-treinada no
ImageNet, congelada, entrada de 224 px no lado menor, e duas cabeças
lineares (cena com softmax e rótulo parcial do indeterminado; legibilidade
com sigmoide, mascarada em "outros"). Parâmetros: 2.975.796 no total; 3.844
treináveis em extração de características (B2). Venceu a baseline não
profunda por +0,256 na métrica otimizadora (IC 95% de +0,104 a +0,378; D3,
seção 6 do notebook), foi calibrada (E1, seção 7) e tem limiar de rejeição
fixado pela meta da E2 (seção 7).

### 3.2 N1 — Arquitetura I × II

| Critério | II — B aperfeiçoada (construída) | I — A melhorada (comparação) |
| --- | --- | --- |
| Modelos aprendidos | 3: triagem, detector, leitor | 5: triagem, 2 detectores, 2 leitores |
| Rótulos de cena e legibilidade | 300 fotos (feito, C2) | Os mesmos 300 |
| Caixas | Todas as fotos de medidor legível, num só detector | O mesmo total, dividido por tipo: com 15% de ciclométricos, cerca de 1.300 fotos para reunir 200 caixas ciclométricas |
| Parâmetros (triagem + detector) | 2,98 M + 3,44 M | 2,98 M + 2 × 3,44 M |
| Parâmetros do leitor | A definir (ET6) | O dobro do da II |
| Latência em CPU | Triagem medida: 7,9 ms por foto (N4); detector e leitor a medir | Igual na triagem; dois detectores e leitores, mas só um roda por foto |

Custo de desenhar caixas (E24): 11,8 s por foto (desvio de 2,5 s; IC 95% de
10,6 a 13,0 s), medido em 20 fotos de medidor legível, depois de um treino
em 5 fotos, com cronômetro automático. Extrapolação: 1.000 fotos custam
cerca de 3,3 h de anotação; as 1.300 fotos que a I exigiria para ter 200
caixas ciclométricas, cerca de 4,3 h. Limitação: um só anotador, já
familiarizado com a base, e sem a transcrição dos recortes.

Escolha: a II primeiro, porque treina cada modelo com todas as fotos e não
desperdiça trabalho: as caixas e os recortes da II servem para treinar a I,
bastando separá-los por tipo (seção 1.2).

### 3.3 N2 — Orçamento de erro da cascata

Taxa de cada estágio para uma foto de medidor legível, listada no
BaseExtracao, chegar a "Leitura confirmada":

| Estágio | Tipo | Taxa | Origem |
| --- | --- | --- | --- |
| Integridade | Regra | 1,00 (limite inferior 0,91) | Medida: 0 de 34 fotos com conteúdo descartadas na calibração do limiar (E16) |
| Pareamento | Regra | 1,00 | Por definição: a foto listada é pareada pelo nome exato (V3) |
| Triagem (não rejeitar um legível) | Aprendido | 1,00 (limite inferior 0,98) | Medida: 0 de 146 medidores legíveis rejeitados, fora do lote (E2) |
| Detector | Aprendido | 0,95 | Hipótese de trabalho, sem medição; a medir na ET5 |
| Leitor | Aprendido | 0,90 | Hipótese de trabalho, sem medição; a medir na ET6 |
| Validadores | Regra | Não reduzem acerto | Só retiram fotos do caminho de aprovação (E10) |

Produto ponta a ponta: 0,95 × 0,90 = 0,855 com as taxas pontuais; 0,91 ×
0,98 × 0,95 × 0,90 ≈ 0,76 com os limites inferiores dos estágios medidos.

Qual estágio mais perde: a resposta depende do tipo de perda.

- **Perda silenciosa** (a foto sai do fluxo sem ninguém ver): só a
  integridade e a triagem podem causá-la, e as duas foram medidas perto de
  zero.
- **Perda de cobertura** (a foto vai para "Verificar"): concentra-se no
  detector e no leitor. Um erro deles gera divergência com o CSV e custa
  tempo do analista, não uma fatura errada.

O ramo "pode melhorar?" das arquiteturas originais foi removido, por não ter
critério de parada (seção 1.1). As fotos que não seguem adiante têm destino
definido: "Rejeitadas / Outros" ou "Rejeitadas / Baixa qualidade" (triagem)
e "Necessita de verificação" (detector, leitor e validadores).

### 3.4 N3 — Especificação do detector

| Item | Especificação |
| --- | --- |
| Modelo (perfil CPU) | SSDlite320 com MobileNetV3-Large (`torchvision`): 3,44 M de parâmetros, 0,58 GFLOPS |
| Candidato (perfil GPU) | Faster R-CNN com MobileNetV3-Large-FPN (`torchvision`) |
| Classes | Display digital, display ciclométrico e placa (número do medidor). A função aparece só no display digital, nunca no ciclométrico; por isso é lida no recorte do display digital, sem classe própria de detecção (decisão final na ET5) |
| Resolução | Entrada de 320 × 320 px, a partir da foto nativa de 360 × 480. O detector só localiza: a leitura é feita no recorte da foto original, porque a redução deixaria cada dígito com cerca de 7 px (guia, seção 4) |
| Licença | BSD-3 |

Consequência da licença do YOLO: o Ultralytics YOLOv8 e o YOLO11 são
AGPL-3.0. Embarcá-los num produto entregue ao cliente obrigaria a distribuir
o código do produto sob a mesma licença, ou a comprar a licença comercial.
A alternativa permissiva adotada é o detector do `torchvision` (E07). O
PP-OCR det (Apache-2.0) também seria permissivo, mas exigiria outro
framework além do PyTorch. O custo da troca é uma precisão de referência
menor (COCO: 21,3 de mAP do SSDlite contra 37,3 do YOLOv8n), a medir na base
do cliente na ET5.

### 3.5 N4 — Implantação local em Windows

| Medida | Resultado |
| --- | --- |
| Paridade PyTorch × ONNX (16 fotos reais) | Maior diferença de logit 1,05 × 10⁻⁵ (< 10⁻⁴); decisões de rejeição idênticas |
| Latência, só o modelo (p50 / p95) | ONNX Runtime 3,3 / 4,2 ms; PyTorch 12,2 / 14,9 ms |
| Latência com leitura e pré-processamento | ONNX Runtime 7,9 / 9,6 ms; PyTorch 17,0 / 21,7 ms |
| Arquivo ONNX | 12,2 MB |

Medido em CPU, com 4 threads (número de núcleos do piso da E04), na máquina
de desenvolvimento (i7-12700H). Os núcleos dela são mais rápidos que os de
uma máquina de escritório antiga: no piso real, os tempos podem ser algumas
vezes maiores. A medição no hardware mínimo fica para a ET9.

A cascata cabe na janela? A triagem de um lote de 3.100 fotos leva menos de
1 minuto. Mesmo uma janela de 1 hora deixaria cerca de 1,2 s por foto para o
detector e o leitor. O motor de inferência do produto (PyTorch ou ONNX
Runtime) é decidido na ET9, com os três modelos medidos.

Achado: com o ONNX Runtime, o pré-processamento (abrir o JPEG, autocontraste
e redimensionamento) passa a custar mais que a rede (cerca de 4,6 dos 7,9
ms).

Achado de MLOps: as características do treino tinham sido extraídas na GPU
com TF32, que arredonda as convoluções. A diferença para a inferência em CPU
chegava a 0,02 de logit, o dobro da margem entre o limiar escolhido e o
medidor legível de maior escore (3,7 × 10⁻⁴ no escore). A extração passou a
ser feita em precisão cheia, e os resultados foram recalculados; a
conferência ponta a ponta caiu para 6,7 × 10⁻⁶.

### 3.6 V1 — Três status viram arquitetura

| Status | Produzido por | Aprendido ou regra |
| --- | --- | --- |
| Leitura confirmada | Triagem deixa passar, detector localiza, leitor lê a sequência esperada do CSV, validadores não objetam | Triagem, detector e leitor aprendidos; confirmação e validadores são regra |
| Divergência ("Necessita de verificação") | O leitor não confirma a sequência esperada, ou um validador objeta (função diferente de 03, criticidade A, fora dos Parâmetros) | Comparação com o CSV e validadores são regra |
| Impedimento ou inconclusivo ("Rejeitada") | Integridade (corrompida), pareamento (não listada), triagem (outros ou baixa qualidade) | Integridade e pareamento são regra; triagem é aprendida |

Comparar o valor lido com o valor digitado não é tarefa de rede: é uma regra
determinística (guia, seção 2).

Estágios × ponta a ponta: não existe rótulo final (nenhuma decisão dos
analistas foi registrada), mas é possível rotular cada estágio. Os estágios
também dão endereço aos erros (seção 3.3) e permitem trocar um modelo sem
mexer nos outros. Uma rede ponta a ponta precisaria de pares (foto, status)
que não existem, e leria a foto inteira reduzida, com dígitos de cerca de 7
px. Escolha: estágios.

### 3.7 V2 — A meta em forma de número

**Meta: precisão ≥ 99,3% em "Leitura confirmada"**, ou seja, risco abaixo
da taxa de erro das anotações de campo, medida em 0,7% (1 em 153 leituras
comparáveis; IC 95% de 0,1% a 3,6%; E23). O software não pode aprovar com
mais erro que o processo atual, porque cada leitura errada aprovada gera
devolução em dobro do valor cobrado a mais (E09, princípio 3).

Por que a meta é alcançável. O software não lê livremente para aprovar: ele
confere a leitura com o valor digitado (E07). Uma leitura só é confirmada
errada se a anotação de campo estiver errada **e** o leitor reproduzir
exatamente o mesmo erro. Nos outros casos, a divergência leva a foto para
"Verificar". Logo:

risco do software ≤ 0,7% × P(o leitor reproduz o erro digitado)

O risco é, por construção, uma fração da baseline, qualquer que seja o
valor real dela dentro do intervalo medido.

Por que não há meta de cobertura. A fração de fotos legíveis varia com a
pessoa, a época e o motivo da foto (E09, princípio 4). No Lab02, a cobertura
da rejeição já variou de 31% a 50% entre os três lotes, com o mesmo modelo e
o mesmo limiar (seção 7 do notebook). A cobertura é maximizada e reportada
por lote, com intervalo, e traduzida em horas de analista economizadas.

O que dá para provar. Para demonstrar precisão ≥ 99,3% pelo limite superior
de 95%, com zero erros, são necessárias 427 leituras aprovadas. O lote de
teste tem cerca de 40, o que só garante risco abaixo de cerca de 7,5%
(E23). A meta está fixada; verificá-la exige ampliar a rotulagem.

Barreiras independentes do modelo: clientes de criticidade "A" nunca são
aprovados automaticamente, e a validação por Parâmetros envia para
"Verificar" as leituras fora da faixa. Essas regras só retiram fotos do
caminho de aprovação, nunca aprovam (E10).

### 3.8 V3 — Pareamento antes do rótulo

Regra: uma foto é pareada a uma linha do BaseExtracao quando o nome do
arquivo coincide exatamente, inclusive maiúsculas e minúsculas, com o nome
listado na linha (E16). Fotos sem linha correspondente são órfãs e recebem
o status "Rejeitada - Não listada em BaseExtracao", sem passar pela triagem.

Nos 4 lotes: 12.340 imagens, 11.474 pareadas e 866 órfãs (7,0%); nenhuma
linha lista uma foto inexistente. Numa amostra aleatória de 30 órfãs,
nenhum número de medidor lido na foto aparece em algum CSV: as órfãs não
são fotos extras de leituras listadas, e rejeitá-las não perde leituras
verificáveis. As órfãs não entram no teste da triagem, porque em produção
são rejeitadas antes dela (E11).

### 3.9 V4 — Os zeros como teste difícil

Os 36 zeros sem nota fora do teste foram usados como conjunto de estresse,
com o limiar da E2 aplicado sem ajuste (seção 7 do notebook):

- 0 de 20 medidores legíveis rejeitados;
- as 3 fotos legíveis com leitura divergente do zero registrado seguem para
  o leitor e o verificador; detectar a divergência é tarefa da ET5 a ET7;
- 8 de 15 medidores ilegíveis e a única foto de "outros" foram rejeitados.

Achado de dados: 43% dos medidores entre os zeros são ilegíveis (15 de
35), contra 29,8% na amostra aleatória (seção 7 do notebook). Um zero sem nota costuma vir de uma foto
que não permitia a leitura. Os 14 zeros do lote de teste seguem lacrados até
a ET4.

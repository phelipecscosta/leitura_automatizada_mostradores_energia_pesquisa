# Detector de campos (exploração do produto)

Estudo que decide o detector do produto, a segunda porta do indeterminado e a
regra do tipo (spec v16, 2.5, itens 10.1, 10.2 e 12; E07, E50, E52, E74, E75).
Não é entrega de rubrica (instruções, 6.2).

## 1. Protocolo de avaliação

Fixado em 17/11/2026, antes de qualquer treino dos candidatos. Mudanças só por
revisão datada (seção 1.12); o horário do commit é a prova. Os códigos entre
parênteses (D1 a D6, C1 a C4) são as decisões da ET5b.

### 1.1 Dados e gabarito

- Amostra: `amostra_caixas_v2.csv`, na pasta de trabalho (SHA-256
  `5423bbdace93f225f2f9d2cc0b6980b871fae140ca52e5bdaffcef4d95244281`): 3.000
  fotos de desenvolvimento e 1.800 do teste 03/07, tiradas das fotos pareadas
  que passam pela triagem C4 (E75). A v2 amplia os determinados do teste em
  600 fotos pela mesma ordem fixa da v1 (D6 b), antes de qualquer leitura do
  teste pelo detector.
- Gabarito: as caixas, em `caixas.csv` (SHA-256
  `0c79825ba04b56dce92213977579dbb9c64eb3a84ee72fa138f6b2f70e53b451`) e
  `caixas_fotos.csv` (SHA-256
  `d1e73df99d4bf3e52f580999014961f006cf482408d124a58f1c867c2a50b8be`), lidas
  e validadas por `training.box_labels`. Esquema: 02_dataset.md, seção 5.
- Pesos amostrais: os da amostra v2 (população do estrato ÷ fotos sorteadas).
  Todas as métricas usam esses pesos; os estratos são os do desenho, e não
  mudam com a correção de rótulos.
- Rótulos de grupo, para o Validador 2 e os relatórios por grupo: R4 no
  desenvolvimento e R5 no teste (`training.group_dataset`). A R5 corrige 5 das
  7 fotos novas do teste apontadas pela conferência das caixas, revisadas às
  cegas, como na E77.
- Foto com campo (D3 a ii): ao menos uma caixa, de qualquer classe, com os dois
  lados ≥ 4 px. As 51 caixas do desenvolvimento com algum lado abaixo de 4 px
  são de medidores muito distantes, mantidas no gabarito; perdê-las não perde
  informação, porque o leitor também falharia.

### 1.2 Partição

- Validação por lote deixado de fora nas 3.000 fotos de desenvolvimento (E75,
  A2 a): 3 dobras; predições fora da dobra reunidas.
- 3 sementes por candidato e dobra.
- Divisão interna, só para escolher a época: 10% das fotos dos lotes de treino
  de cada dobra, com semente fixa, a mesma para todos os candidatos. O lote
  avaliado nunca participa.
- Teste 03/07: leitura única, com o candidato, o limiar e a regra do tipo
  congelados (seção 1.10).

### 1.3 Candidatos (C1 b, C4 a)

| Código | Modelo | Perfil | Entrada (largura × altura) |
| --- | --- | --- | --- |
| S1 | SSDlite + MobileNetV3-Large, 6 níveis (passo 16 em diante) | CPU | 360 × 480, nativa, sem achatar |
| F1 | Faster R-CNN + MobileNetV3-Large-FPN, 5 níveis (passo 8 em diante, N8) | CPU | 360 × 480, nativa |
| F2 | O mesmo do F1 | GPU | 600 × 800, ampliada |

- Código: `meter_reader.detector` (montagem, também usada pelo produto) e
  `training.detector_anchors` (formas e latência), do `torchvision` (BSD-3).
- S1: a montagem do `ssdlite320_mobilenet_v3_large` com outro tamanho e outras
  âncoras; a função pronta fixa 320 × 320.
- F1 e F2: a FPN recebe os estágios de passo 8, 16, 32 e 32 da MobileNetV3
  (N8), mais o nível extra. O padrão do `torchvision` usa só o passo 32, porque
  amplia a foto para 800 px de lado.
- F1: 150 propostas da RPN na inferência, como a versão de baixa resolução do
  `torchvision` (`fasterrcnn_mobilenet_v3_large_320_fpn`); F2: o padrão (1.000).
- Âncoras (C4 b, revisada em 1.12, R1): k-means das formas (largura, altura) das
  caixas do desenvolvimento, com distância 1 − IoU, na escala da entrada de cada
  candidato, repetidas em todos os níveis. Formas por posição: 6 no S1 e 15 no
  F1 e no F2, as mesmas da configuração-padrão. Meta: em cada classe, ≥ 90% das
  caixas com alguma âncora de IoU ≥ 0,5. As formas ficam em
  `detector_ancoras.json`, na pasta de trabalho.
- Fora: FCOS (só com a ResNet-50 no `torchvision`); D-FINE (dependência nova,
  licença não conferida, sem evidência de necessidade); YOLO (licença, E07).
- Funções internas do `torchvision` usadas na montagem (prefixo `_`) estão
  protegidas pela versão travada (0.29).

### 1.4 Pré-treino (C2 iii)

- Padrão: espinha pré-treinada no ImageNet (`IMAGENET1K_V1`, a da referência de
  detecção do `torchvision`); o resto da rede começa do zero.
- Ablação, só no vencedor: as camadas compatíveis dos pesos COCO do
  `torchvision` para a mesma arquitetura. Como as âncoras, a cabeça e, no
  Faster R-CNN, a FPN mudam, só a espinha e as camadas de mesma forma são
  carregadas; o protocolo relata quais.
- Regra da ablação: adotada se a diferença de AP ponderada (seção 1.6) tiver o
  IC 95% inteiro acima de zero e o candidato continuar viável.
- Licença dos pesos: o código é BSD-3, mas os pesos pré-treinados do
  `torchvision` não declaram licença. É a mesma incerteza da triagem C4
  (E28), a confirmar antes da ET9.

### 1.5 Treino (C4 c)

- Entrada: o mesmo realce fixo da triagem (autocontraste de 1%), no treino e no
  produto (E19, E33), em [0, 1]; a normalização é a do próprio modelo.
- Aumentação: brilho, contraste, translação e escala de ±10%, com as caixas
  transformadas junto. Sem espelhamento (inverte os dígitos) e sem rotação
  (alarga as caixas retas, Q-d).
- SGD com momento de 0,9, aquecimento e decaimento em cosseno; lote de 8,
  descartando o lote final incompleto (o último nível do SSDlite tem 1 × 1
  posição, e a normalização em lote não treina com uma foto só).
- Até 40 épocas; a época escolhida é a de maior AP ponderada na divisão
  interna. O modelo final, com os 3 lotes, para na mediana das épocas
  escolhidas, como na triagem (R5 da triagem).
- Hiperparâmetros de prática usual, sem medição nesta base (como na E57).

### 1.6 Métricas de localização (D1 c, D2 a, D4 a)

- Acerto: caixa prevista da mesma classe com IoU ≥ 0,5 com a caixa anotada
  (padrão do PASCAL VOC), uma caixa prevista por anotada, da maior confiança
  para a menor.
- AP ponderada por classe: a curva de precisão × revocação com cada caixa
  (anotada e prevista) pesando o peso da sua foto, com 101 pontos de revocação
  (forma do COCO).
- Métrica de escolha: a média das AP ponderadas de `display_digital`,
  `display_ciclometrico` e `numero_serie`. O `display_kvarh` fica fora da média;
  a confusão kvarh → display de consumo é relatada.
- Relatada à parte: a cobertura do campo, a fração da caixa anotada dentro da
  caixa prevista (p5 e mediana por classe). Ela dimensiona a margem do recorte
  na ET6.
- Limites superiores de proporções ponderadas: Clopper-Pearson com o tamanho
  efetivo de Kish (taxa × tamanho efetivo no papel do número de eventos; Korn e
  Graubard, 1998, a verificar), em `training.detector_eval`.

### 1.7 Segunda porta (D3)

- Escore da foto: a maior confiança entre as caixas detectadas, de qualquer
  classe. Rejeita-se a foto com escore abaixo do limiar t.
- Perda: soma dos pesos das fotos com campo rejeitadas ÷ soma dos pesos das
  fotos com campo. Meta: ≤ 0,5% pelo limite superior de 95% (D3 b 1).
- Cobertura: fração ponderada das fotos sem campo rejeitadas.
- Limiar: entre os que cumprem a meta nas predições fora da dobra, o de maior
  cobertura (D3 c β). O corte das caixas entregues ao leitor fica para a ET6.
- Viável: o candidato para o qual algum limiar cumpre a meta.
- Relatado: a soma das perdas da triagem e da segunda porta, no
  desenvolvimento.

### 1.8 Regra do tipo, Validador 2 (D5)

- Família vista pelo detector, entre as caixas acima de t: digital (algum
  `display_digital`), ciclométrico (algum `display_ciclometrico` ou
  `display_kvarh`) ou nenhuma (só `numero_serie`).
- As duas famílias na foto: decide a família do display de maior confiança
  (A2).
- Só número de série: segue sem comparação (B1); o caminho dessa foto é da ET6.
- Família diferente do tipo da triagem: "Verificar", com o motivo "Tipo
  divergente entre triagem e detector".
- Relatado: quantas fotos têm as duas famílias e, entre elas, em quantas o
  display de maior confiança diverge da triagem.

### 1.9 Regra de escolha (C3)

- Ordem de simplicidade: latência p95 em CPU (4 threads, lote de 1), medida na
  seção 1.11 antes do treino: S1, depois F1.
- Perfil CPU (C3 a 1): entre os candidatos viáveis com p95 ≤ 200 ms na máquina
  de desenvolvimento (L1), vence o mais simples cuja AP ponderada fique dentro
  de 5 pontos (Δ3, escala de 0 a 100) da melhor, pelo limite inferior do IC 95%
  da diferença (bootstrap pareado por foto, 2.000 reamostragens).
- Perfil GPU (C3 d, G2): o F2 só é entregue se superar o vencedor do perfil CPU
  em pelo menos 10 pontos de AP ponderada, pelo limite do IC, e for viável.

### 1.10 Teste 03/07 (D6)

- 1.723 fotos com campo (tamanho efetivo de Kish de 1.718) e 77 sem campo:
  orçamento de 3 perdas para a meta de 0,5%. O orçamento foi calculado só com o
  gabarito e os pesos, sem nenhuma predição.
- Leitura única: perda e cobertura da segunda porta, AP ponderada por classe,
  cobertura do campo, regra do tipo e latência p95 do modelo final em CPU.
- É a primeira leitura do 03/07 pelo detector (E75). A triagem já leu esse lote
  na E68; a população do detector vem dos logits gravados naquela abertura.

### 1.11 Medições antes do treino

Geometria das caixas do desenvolvimento (T5b.1), em pixels da foto original de
360 × 480:

| Classe | Caixas | Largura p5 / p50 / p95 | Altura p5 / p50 / p95 | Proporção p50 |
| --- | --- | --- | --- | --- |
| display_digital | 2.193 | 16,5 / 108,8 / 221,3 | 6,0 / 37,7 / 80,3 | 2,91 |
| display_ciclometrico | 715 | 25,8 / 80,9 / 152,4 | 7,7 / 22,5 / 47,7 | 3,77 |
| display_kvarh | 179 | 41,9 / 73,9 / 125,3 | 12,5 / 24,4 / 43,6 | 3,58 |
| numero_serie | 2.207 | 36,0 / 80,6 / 134,6 | 8,3 / 17,2 / 32,6 | 4,77 |

Fração das caixas com alguma âncora de IoU ≥ 0,5 (digital / ciclométrico /
kvarh / número de série) e latência em CPU, sem pesos treinados:

| Configuração | Cobertura das âncoras | Latência p50 / p95 |
| --- | --- | --- |
| Padrão: SSDlite 320 × 320 | 53% / 21% / 15% / 5% | — |
| Padrão: Faster R-CNN FPN, 800 × 1066 | 76% / 55% / 68% / 22% | — |
| Formas divididas por nível (primeira versão): S1 | 13% / 18% / 12% / 15% | 25,4 / 31,4 ms |
| Formas divididas por nível (primeira versão): F1 sem N8 | 47% / 42% / 42% / 36% | 58,7 / 68,0 ms |
| S1, formas repetidas | 81% / 81% / 80% / 77% | 25,2 / 31,5 ms |
| F1, formas repetidas, N8 | 91% / 95% / 97% / 98% | 88,5 / 102,4 ms |
| F2, formas repetidas, N8 | 94% / 97% / 97% / 99% | 332,3 / 358,6 ms |

O S1 não alcança a meta de 90% das âncoras, limitado pelo passo 16 do primeiro
nível; o valor alcançado fica registrado, e a regra da seção 1.9 decide.

### 1.12 Revisões

Revisão de 17/11/2026, antes de qualquer treino.

**R1 — Âncoras repetidas em todos os níveis e N8 (seção 1.3).** A primeira
versão da C4 b dividia as formas, ordenadas por área, por igual entre os
níveis. Ela cobriu menos que as âncoras-padrão (tabela da seção 1.11), porque as
formas grandes caíam em grades de passo grosso, de 64 a 512 px. As formas
passam a ser repetidas em todos os níveis, como o `torchvision` faz no Faster
R-CNN. Na mesma medição, o Faster R-CNN da MobileNetV3 mostrou só mapas de passo
32 na resolução nativa; a FPN passa a receber também os de passo 8 e 16 (N8). A
mudança usa só a geometria do gabarito do desenvolvimento, sem nenhum treino.


Revisão de 17/11/2026, antes de qualquer treino.

**R2 — Sementes, números do treino e precisão (seções 1.5, 1.7 e 1.9).**

- Sementes (T1 a): cada semente é avaliada separadamente. A métrica de escolha
  é a média das AP ponderadas das 3 sementes; o bootstrap pareado sorteia as
  fotos uma vez por reamostragem e recalcula a média das 3, de modo que o IC
  inclui o ruído entre sementes. O limiar da segunda porta é o menor dos 3
  limiares escolhidos, um por semente (lado seguro). O produto usa um modelo
  só: o final da semente 0, treinado com os 3 lotes. Juntar as caixas das
  sementes exigiria fusão de caixas e triplicaria a latência.
- Números do treino (T2): as receitas de referência do `torchvision` na regra
  linear para o lote de 8. S1: taxa de aprendizado de 0,00625 (referência de
  0,15 com lote de 192) e decaimento dos pesos de 4 × 10⁻⁵, com todas as
  camadas da espinha treináveis. F1 e F2: 0,01 (referência de 0,02 com lote de
  16) e 10⁻⁴, com as 3 últimas camadas da espinha treináveis. Aquecimento de 1
  época. A referência do SSDlite treina 600 épocas, contra as 40 daqui: a
  escolha da época protege contra o sobreajuste, mas não contra o subtreino.
- Precisão (T3 a): treino em precisão mista na GPU, como na triagem (E62);
  predição da divisão interna e fora da dobra em precisão completa, sem TF32
  (E33).


Revisão de 17/11/2026, **depois** dos resultados do desenvolvimento (declarada).

**R3 — Contabilidade da segunda porta e corte de confiança (seções 1.3 e 1.7).**

- Resultado que a motivou: no F1, o limiar escolhido foi 0 nas 3 sementes, com
  perda e cobertura nulas. Em 1,0% a 1,7% das fotos com campo, o F1 não entregou
  nenhuma caixa (103, 79 e 125 fotos por semente; 30 nas três, com mediana de
  16 px no menor lado do maior campo). Essas fotos "passavam" pela porta e não
  contavam como perdidas, mas sem caixa o leitor não tem recorte para ler.
- R3 a: a foto com campo sem nenhuma caixa entregue conta sempre como perdida;
  o menor limiar é o menor número positivo. É uma correção da definição da
  seção 1.7, que não descrevia o que acontece no pipeline.
- R3 b: o mesmo corte de confiança da inferência, 0,001, em todos os
  candidatos. O S1 já usava 0,001 (padrão do SSDlite); o F1 e o F2 usavam 0,05
  (padrão do Faster R-CNN do `torchvision`), que descarta as caixas de
  confiança baixa antes da porta e do leitor. A diferença era de configuração,
  e não do que os modelos aprenderam.
- Consequência: o F1 é retreinado (9 execuções, mesmas sementes e
  configuração), porque só as predições tinham sido gravadas. O treino passa a
  gravar os pesos de cada execução. As partes antigas do F1 ficam guardadas na
  pasta de trabalho, como registro.
- O S1 não muda: o corte dele já era 0,001. O F2 não é retreinado: está a 2,4
  pontos de AP do F1 (IC inferior de +2,0), contra os 10 da regra G2, e o corte
  afeta sobretudo a porta, e não a AP. Se o F1 continuar não viável depois da
  R3, o F2 também é retreinado (B+).
- O teste 03/07 continua lacrado. A regra da seção 1.9 é reaplicada sem
  nenhuma outra mudança.

### 1.13 Limitações declaradas

- Um único anotador das caixas, sem medida de concordância, e anotação não cega
  ao grupo e à condição (02_dataset, 5.6).
- As 600 fotos novas do teste foram anotadas numa sessão separada, depois da
  medição da geometria; a ferramenta e as regras foram as mesmas.
- `display_kvarh` com 179 caixas no desenvolvimento e 30 no teste.
- O teste só tem 77 fotos sem campo: a cobertura da segunda porta sai com
  intervalo largo.
- A latência foi medida na máquina de desenvolvimento, e não no piso da E04.


# Leitor dos campos (exploração do produto)

Estudo que decide o leitor do produto, a leitura com verificação e as três faixas
de confiança (spec v18, 2.5, itens 10.1, 11, 12 e 13; E07, E35, E50, E53, E96).
Não é entrega de rubrica (instruções, 6.2).

Pergunta da etapa (ET6b): qual configuração do leitor vai para o modelo final da
ET6c, e com quais limiares das três faixas, pelas predições fora da dobra por lote
deixado de fora, avaliadas pelas regras desta seção.

## 1. Protocolo de avaliação

Fixado em 20/11/2026, antes de qualquer treino do leitor. Mudanças só por revisão
datada (seção 1.14); o horário do commit é a prova. Os códigos entre parênteses
são as decisões da ET6b (T6b.1: Z, T, V, F, D, O; T6b.2: P1 a P8).

### 1.1 Dados e rótulos

- Rótulos: `rotulos_leitor.csv`, na pasta de trabalho (SHA-256
  `577848233EA1EB76A99BE3DEDA8D433AD09B0C764D1611B8E873B6B34CF8BE13`; o valor completo fica fixado no carregador do `training`).
  Esquema e origem: 02_dataset.md, seção 6.
- Alvos de treino: status `aceito` ou `humano` (4.077 recortes: 1.848 digitais,
  348 ciclométricos e 1.881 números de série). `parcial_humano` e
  `ilegivel_humano` não são alvo de transcrição.
- Os 52 rótulos `sonnet_csv` (ciclométricos) entram no padrão; a versão sem eles
  é uma ablação (seção 1.9).
- Recortes: `meter_reader.crops`, com a margem por classe (E92), a partir da foto
  aberta sem a orientação EXIF.
- O lote de cada foto vem dos rótulos de grupo R4 do desenvolvimento
  (`training.group_dataset`). Todos os alvos têm lote.

### 1.2 Partição (P1)

- Validação por lote deixado de fora: 3 dobras; predições fora da dobra reunidas.
- 3 sementes (0, 1 e 2) por configuração e dobra.
- Divisão interna, só para escolher a época: 10% dos recortes dos lotes de treino
  de cada dobra, por foto, com semente fixa, a mesma para todas as configurações.
  O lote avaliado nunca participa.
- O teste 03/07 não é usado nesta etapa, nem para pseudo-rótulos. A leitura única
  dele é feita na ET6c (A1-b, E96).

### 1.3 Alvos

| Campo | Alvo | Decisão |
| --- | --- | --- |
| Leitura digital | O texto exibido, com os zeros à esquerda | Z-a |
| Leitura ciclométrica | O texto exibido; no tambor em transição, a alternativa igual à leitura do CSV | T-a |
| Número de série | O texto, só letras maiúsculas e dígitos | V-a |
| Função (digital) | O código como exibido, completo, inclusive o combinado YY(Y)XX; vazio se o display não mostra código | F-c |
| Duas leituras (digital) | A marca do professor | D-b |

- Alfabeto da cabeça do texto principal: 0 a 9 e A a Z, mais o branco da CTC.
  Alfabeto da cabeça da função: 0 a 9, mais o branco.
- Comparação com o CSV: por número inteiro na leitura (`0012345` = `12345`,
  spec 13.2) e por texto no número de série (bloco mais à esquerda do CSV).
- Displays de duas leituras: o recorte inteiro treina só a cabeça "duas
  leituras" (as perdas de CTC ficam desligadas nele); a faixa de 0 a 55% da
  altura do recorte treina as cabeças de CTC, com o alvo da leitura de cima. Na
  inferência, quando a cabeça dispara, o leitor relê essa faixa (D-b).
- A cabeça "duas leituras" também treina com a marca do professor nos recortes
  digitais sem rótulo de texto, com as perdas de CTC desligadas.

### 1.4 Entrada e pré-processamento II (P7, PP-b)

- RGB com o realce fixo (autocontraste de 1%, `meter_reader.preprocess.enhance`)
  aplicado ao recorte.
- Altura de 48 px, mantendo a proporção; largura máxima de 384 px (8 × 48), com
  redução proporcional quando o recorte é mais largo; preenchimento à direita.
- Normalização pelo ImageNet nos dois candidatos.
- Orientação (O-b reduzida): o recorte é lido a 0° e a 180°, e vale a rotação
  com maior confiança da leitura livre (seção 1.6). O treino usa só os recortes
  em pé.

### 1.5 Candidatos (P5)

Estrutura comum: espinha convolucional; as faixas de altura que sobram são
concatenadas por coluna (a posição vertical é preservada, F-c); BiLSTM de 2
camadas com 256 unidades por direção; três cabeças:

- CTC do texto principal (leitura ou número de série);
- CTC da função (perda só nos digitais);
- "duas leituras" (média das saídas da BiLSTM, uma saída; perda só nos digitais).

Um único modelo para as três classes (S0, E53).

| Código | Espinha | Passo (altura × largura) |
| --- | --- | --- |
| L1 | CNN própria, treinada do zero: 6 blocos de convolução 3 × 3, normalização em lote e ReLU (32, 64, 128, 128, 256, 256 canais) | 16 × 4 (sobram 3 faixas) |
| L2 | Prefixo da MobileNetV3-Small, pesos ImageNet do `torchvision`, até o passo 8, com o último passo horizontal trocado por 1 | 8 × 4 (sobram 6 faixas) |

- Código do modelo no produto (`meter_reader`), usado também pelo `training`.
- Latência medida em CPU, 4 threads, lote de 1 recorte, com o
  pré-processamento, em 200 recortes fixos (permutação com semente 0); teto de
  50 ms no p95 por recorte.

### 1.6 Leitura com verificação e confiança

- Leitura livre: decodificação gulosa da CTC (melhor caminho, repetições
  colapsadas, brancos removidos).
- Confiança de uma sequência: a probabilidade dela pela CTC (soma sobre os
  alinhamentos, `ctc_loss` sem redução).
- Confiança da leitura livre: a confiança da sequência decodificada.
- Probabilidade do valor do CSV (verificação): na leitura, o máximo entre o valor
  do CSV completado com 0 a k zeros à esquerda, até 8 caracteres (Z-a); no número
  de série, o bloco do CSV.
- Função: a confiança de cada código pela cabeça da função; a interpretação
  (consumo, YY(Y)XX, ausência de código) é do validador 13.3, na ET6c.
- Temperatura: uma por cabeça de CTC, ajustada nas predições fora da dobra com
  validação aninhada por lote (como na E31), só para que o índice gravado seja uma
  probabilidade. Os limiares não dependem dela. Relata-se o ECE por sequência.

### 1.7 Risco: confirmação falsa (P3, R-c, Y-b)

- Para cada alvo de leitura (digital e ciclométrico) e de número de série, dois
  impostores, com semente fixa: um com um caractere trocado em posição e valor
  sorteados (uniforme); outro com um dígito trocado por um par confundível (0↔8,
  3↔8, 5↔6, 6↔8, 9↔8, 1↔7). Na leitura, a troca é feita no valor numérico, para
  que o impostor nunca seja equivalente ao verdadeiro.
- Um impostor é confirmado quando a leitura livre é igual a ele e a probabilidade
  dele é de pelo menos o limiar alto.
- Limiar alto: um por classe (digital, ciclométrico, número de série), o menor que
  mantém o limite superior unilateral de 95% (Clopper-Pearson) da taxa de
  confirmação dos impostores confundíveis em no máximo 5%. A taxa dos uniformes é
  relatada.
- Casos reais: os recortes da F-2 em que o julgamento humano diverge do CSV,
  inclusive as outras funções, são relatados à parte.
- Relação com a E35: o risco do sistema é no máximo 0,7% × a probabilidade de o
  leitor confirmar um valor errado; a confirmação falsa estima esse segundo fator.

### 1.8 Métricas (P2)

- Escolha (M-c): cobertura no ponto de operação, a média simples das três
  classes. Cobertura de uma classe: fração dos alvos cuja leitura livre é igual
  ao rótulo e cuja probabilidade é de pelo menos o limiar alto da seção 1.7.
- Relatada (M-a): acerto exato da sequência, a média simples das três classes.
- IC: bootstrap pareado por recorte, estratificado por classe, 2.000
  reamostragens.
- Função: acerto exato; matriz de confusão entre 03, 003, 103, 24, 55 e vazio.
  Gatilho da F-b (classificação da função como ablação): acerto exato abaixo de
  95%, ou mais de 2% dos 103 lidos como 03 ou 003.
- Duas leituras: AP e revocação no limiar de 0,5.
- Tudo por classe e por lote.

### 1.9 Regra de escolha e ablações (P5, PP-d)

1. L1 × L2: vence o mais rápido (p95) se o limite inferior do IC de 95% da
   diferença de M-c (mais rápido − mais lento) ficar acima de −3 pontos (Δ = 3);
   senão, vence o outro. Um candidato acima do teto de latência sai.
2. Sobre o vencedor, uma variável por vez, nesta ordem, cada uma partindo da
   configuração adotada até ali:
   1. Autotreinamento (seção 1.10): adotado se a diferença de M-c tiver o IC
      acima de zero.
   2. Cinza (PP-a, 1 canal repetido em 3 no L2): substitui o RGB se a diferença
      de M-c tiver o IC acima de zero.
   3. Sem `sonnet_csv`: os 52 rótulos saem se a diferença de M-c tiver o IC
      acima de zero.
   4. Leitor por família (E53): um modelo só para ciclométricos, comparado ao S0
      nos ciclométricos; adotado se o ganho de cobertura dos ciclométricos for de
      pelo menos 5 pontos, com o IC acima de zero.

### 1.10 Autotreinamento (P6, S-a)

- Na dobra k, os candidatos a pseudo-rótulo são os recortes dos lotes de treino
  dessa dobra sem alvo de transcrição (leitura e número de série), lidos pelo
  modelo da dobra k (semente 0) da configuração adotada.
- Aceitação: a leitura livre é igual ao CSV (comparação da seção 1.3) e a
  probabilidade é de pelo menos o limiar alto da classe, calculado (seção 1.7)
  só com as predições fora da dobra dos dois lotes de treino. O lote avaliado e
  o 03/07 nunca recebem pseudo-rótulo.
- Nos recortes pseudo-rotulados, a perda da cabeça da função fica desligada.
- Uma rodada: modelos novos (3 sementes) treinados do zero com os rótulos e os
  pseudo-rótulos.
- S-b: até duas rodadas a mais, só se a primeira ganhar pelo menos 5 pontos de
  cobertura dos ciclométricos, com o IC acima de zero; para quando o ganho de M-c
  tiver o IC incluindo zero.

### 1.11 Faixas e limiar inferior (P4, L-a)

- Alta: a leitura livre é igual ao valor do CSV e a probabilidade dele é de pelo
  menos o limiar alto (seção 1.7). Segue para os validadores.
- Muito baixa: nenhuma leitura extraída, ou a confiança da leitura livre abaixo do
  limiar inferior. Vai para "Falha na extração".
- Intermediária: o resto. Vai para "Verificar".
- Amostra de legibilidade: 300 recortes de display (digital e ciclométrico) do
  desenvolvimento, sorteados com semente em texto entre todos os recortes, de
  qualquer status. Cada um é julgado numa ferramenta em `local/`, com o recorte e
  o valor do CSV: "o recorte confirma este valor?" (sim; mostra outro valor; não
  dá para ler). O julgamento é feito antes de ver as confianças.
- Limiar inferior: o maior que mantém em no máximo 10% (V) a fração de "sim"
  entre os recortes da amostra com confiança abaixo dele, com pelo menos 30
  recortes abaixo; confianças fora da dobra da configuração final da ET6b.

### 1.12 Treino e aumentação (P7)

- AdamW (taxa de 1 × 10⁻³ no L1 e 5 × 10⁻⁴ no L2; decaimento de 1 × 10⁻⁴),
  cronograma cosseno com 1 época de aquecimento, até 60 épocas, lote de 64.
- Lotes com as três classes em partes iguais (B-b), com reposição dentro da
  classe; uma época tem o número total de alvos ÷ 64 passos.
- Perdas somadas com pesos iguais. Época escolhida pela M-a na divisão interna.
- Precisão mista no treino e 32 bits na predição.
- Aumentação, sem espelhamento e sem giro de 180°: rotação de até ±5°; leve
  perspectiva; deslocamento e escala do recorte de até 5% (variação das caixas do
  detector); brilho, contraste e gama; matiz e saturação; cinza em 20% dos
  exemplos; borrão; compressão JPEG.

### 1.13 Gatilho dos cantos e análise de erros (P8, C-a)

- 50 erros de transcrição fora da dobra da configuração final, sorteados com
  semente fixa, proporcionais às classes. Causa julgada numa ferramenta em
  `local/`: inclinação ou perspectiva, borrão, reflexo, oclusão, campo cortado,
  rótulo errado, outro.
- Gatilho (Q-d, E74): inclinação ou perspectiva em pelo menos 30% dos erros leva
  à anotação dos cantos e à retificação, por revisão deste protocolo.
- Campo cortado em pelo menos 10% dos erros é relatado como evidência contra a
  margem da E92.

### 1.14 Revisões

R1 (20/11/2026, antes do primeiro treino). A amostragem por classes iguais (B-b, seção 1.12) vale para os itens com alvo de texto. Os itens que treinam só a cabeça “duas leituras” (marca do professor, seção 1.3) entram num fluxo próprio, com 16 itens sorteados por lote, ao lado dos 64 itens de texto. A época continua definida pelo número de alvos de texto. Motivo: 4.297 itens só da cabeça, contra 1.848 digitais com texto, ocupariam cerca de 70% da parte digital de cada lote.

R2 (20/11/2026, antes do primeiro treino).
(a) Candidatos: L1; L2-9 e L2-12, com as 9 e as 12 primeiras camadas da MobileNetV3-Small, em que os passos que fariam o passo total passar de 8 na altura e 4 na largura são trocados por 1. Vence o mais rápido (p95) cujo limite inferior do IC da diferença de M-c para o melhor fique acima de −3 pontos. Motivo: o prefixo que termina no passo 8 tem só 4 camadas (cerca de 10 mil parâmetros), e a comparação não isolaria o pré-treino; a latência medida mostra que a BiLSTM domina o custo.
(b) Sementes: o limiar alto e a M-c são calculados por semente, e vale a média das três; o bootstrap reamostra os recortes com os limiares fixos. Motivo: o produto usa uma rede.
(c) Orientação: a leitura a 180° só é feita quando a de 0° não fica na faixa alta, e vale a de maior confiança. Motivo: o giro ocorre em cerca de 0,3% dos recortes.

R3 (20/11/2026, antes do primeiro treino). Até 120 épocas (seção 1.12), com o cronograma cosseno no mesmo comprimento; a época continua escolhida pela M-a na divisão interna. Se, em mais da metade dos treinos de uma configuração, a época escolhida estiver entre a 108 e a 120, o resultado é declarado possivelmente subtreinado. Motivo: na sanidade (T6b.4), sair do platô da CTC levou de 1.000 a 1.500 passos no sobreajuste de lotes fixos, e 60 épocas somam cerca de 2.200.

R4 (20/11/2026, antes dos treinos válidos). A entrada do leitor passa a ser 48 × 192 px, redimensionada sem manter a proporção, sem preenchimento (como no CRNN de Shi et al.); com o passo horizontal de 4, a CTC tem 48 passos de tempo. Substitui “altura de 48 px, mantendo a proporção; largura máxima de 384 px, com preenchimento” (seção 1.4). Motivo: o primeiro treino do L1 (dobra 20/05, semente 0) terminou com M-a interna de 0,003; o diagnóstico descartou a precisão mista e a aumentação, e um teste com imagens sintéticas mostrou que a tela com preenchimento (96 passos) não sai do platô da CTC em 400 passos, enquanto a entrada esticada (48 passos) chega a acerto de 1,00 em 750. O treino que falhou foi descartado e não entra em nenhum resultado.

R5 (20/11/2026, antes dos treinos válidos). Só a cabeça do texto treina a espinha e a BiLSTM principal. A função tem um ramo próprio (projeção e BiLSTM de 128 unidades) sobre as características da espinha, sem gradiente para ela; a cabeça de duas leituras usa a saída da BiLSTM principal, sem gradiente. Motivo: com imagens sintéticas, a cabeça da função compartilhada impediu o texto de sair do platô. No diagnóstico com o L2-9 (2.000 passos, primeira dobra), a M-a interna e o acerto da função foram: só o texto, 0,616; viés inicial do branco na função, 0,003 e 0,807; cabeças sem gradiente para o tronco, 0,574 e 0,128; ramo próprio, 0,551 e 0,780. O critério fixado antes desse último teste (texto de pelo menos 0,55 e função de pelo menos 0,70) foi cumprido pelo ramo próprio. Os treinos e diagnósticos anteriores a esta revisão não entram em nenhum resultado.

R6 (20/11/2026, antes dos treinos válidos). Até 240 épocas (seção 1.12, R3), com o cronograma cosseno no mesmo comprimento; o alarme de subtreino da R3 passa a olhar as épocas de 216 a 240. Motivo: no diagnóstico só com o texto (4.000 passos, primeira dobra), o L1 saiu do platô entre os passos 2.500 e 3.000, e o L2-9 perto do passo 1.000; com 120 épocas (cerca de 4.400 passos), o L1 seria comparado sem ter terminado de aprender, e a regra da seção 1.9 não isolaria o efeito do pré-treino (O3).

R7 (21/11/2026, depois das predições fora da dobra dos candidatos, antes da regra de escolha). O leitor lê só a 0 grau (O-a), substituindo a O-b reduzida (seção 1.4) e o item (c) da R2. Motivo: nas predições fora da dobra, a escolha da rotação pela confiança ficou com 180 graus em 12% a 24% dos itens, contra cerca de 0,3% de recortes girados na rotulagem, e reduziu o acerto exato em 1,4 (L1), 3,6 (L2-9) e 3,9 (L2-12) pontos; nos itens trocados, o acerto a 180 graus foi de 0,3% a 1,6%. A rede, sem giros de 180 graus no treino, dá leituras confiantes a recortes invertidos. A mudança vale igualmente para os três candidatos e não altera a regra de escolha; os recortes girados vão para "Verificar" ou "Falha na extração" pela confiança.

R8 (21/11/2026, antes do treino das ablações). O leitor por família (seção 1.9, item 4) é um especialista por ajuste fino. Ele parte dos pesos da configuração adotada, da mesma dobra e da mesma semente, e continua o treino só com os itens de texto ciclométricos dos lotes de treino (e os pseudo-rótulos ciclométricos, se o autotreinamento for adotado). Regras: lotes de 64 ciclométricos; taxa de 1/10 da do candidato; aquecimento de 1 época e cosseno; 60 épocas com os passos por época da configuração adotada; época escolhida pela M-a interna dos ciclométricos, com a época 0 (o modelo de partida) entre as opções. A comparação e a regra de adoção não mudam. Motivo: treinado a partir do ImageNet com cerca de 210 ciclométricos, o especialista partiria sem o que o modelo único aprendeu com os outros cerca de 2.200 recortes; a comparação mediria a falta de dados, e não o ganho de especializar. Incerteza: a divisão interna tem cerca de 25 ciclométricos por dobra, e a escolha da época é ruidosa.

R9 (21/11/2026, antes do treino das ablações). Os 868 recortes ciclométricos dos lotes de desenvolvimento sem alvo de texto (professor: 260 discorda, 122 parcial, 23 erro de formato, 463 ilegível) recebem rotulagem humana às cegas. A leitura é digitada sem ver o CSV; se diferir do CSV, o CSV é mostrado e o rotulador mantém, corrige ou marca ilegível. Regras:

(a) o rótulo é o número sem o tambor decimal (D-a), com uma marca de presença do tambor decimal;
(b) [ab] só quando nenhum algarismo aparece inteiro na janela (TR-a), resolvido pela T-a;
(c) só vira alvo o recorte com todos os tambores legíveis (a transição conta como legível). Não viram alvo: ilegível (algum dígito ilegível por sujeira, reflexo ou desfoque), obstruído (algo à frente dos tambores ou parte deles fora do recorte) e girado (rotação de 45 a 90 graus, ou de 180 graus). Recortes inclinados de 15 a 44 graus são rotulados, com uma marca; abaixo de 15 graus, sem marca. Os rótulos passam a uma nova versão, com SHA-256 próprio. A configuração de partida (L2-12) é retreinada com eles, e os artefatos anteriores dela são arquivados; a escolha do candidato (seção 1.9) não é refeita. Motivo, pelo diagnóstico do L2-12 fora da dobra: o acerto dos ciclométricos cai de 0,919 nos vistos para 0,618 na divisão interna e para 0,474 no lote fora da dobra (variância e mudança entre lotes), e os rótulos aceitos privilegiam os recortes fáceis (acerto de 0,491 no haiku_csv e de 0,378 no sonnet_csv). Consequências: a M-c dos ciclométricos passa a incluir os recortes difíceis e deixa de ser comparável com a da seção 2 anterior a esta revisão. A decisão foi informada por erros fora da dobra, o que torna as comparações seguintes um pouco otimistas; o juiz sem viés é o lote 03/07. Um único rotulador. Os recortes que um humano julgou sem alvo (F-2 e R9) deixam de ser candidatos a pseudo-rótulo (seção 1.10): o julgamento humano prevalece sobre o do modelo.

R10 (23/11/2026, antes do cálculo do limiar inferior e do comitê).

(a) Limiar inferior: calculado separadamente com as confianças fora da dobra de cada semente da configuração final. O adotado é a mediana dos três. A leitura de cada recorte da amostra é a do produto: recorte inteiro e, num digital em que a cabeça de duas leituras dispara, a faixa de cima (D-b). Recorte sem leitura extraída tem confiança −∞ e fica abaixo de qualquer limiar. Se o comitê for adotado, o limiar é recalculado com a confiança do comitê.
(b) Comitê das 3 sementes: para cada recorte, a probabilidade de uma sequência é a média das probabilidades dos 3 modelos da dobra. A leitura livre é, entre as leituras livres dos 3, a de maior probabilidade média. A confiança e a probabilidade do valor do CSV seguem a mesma média. Os limiares altos são recalculados pela regra da seção 1.7. O comitê é comparado com o modelo único (M-c média das sementes) por bootstrap pareado por recorte. É adotado se a diferença de M-c tiver o IC acima de zero e a latência p95 dos 3 modelos em sequência (CPU, 4 threads, lote 1) for de no máximo 50 ms.

R11 (23/11/2026, antes do treino das cabeças). O gatilho da F-b disparou: acerto exato da função de 0,806 e 66% dos 103 lidos como 03 ou 003. A cabeça de duas leituras não disparou nenhuma vez (AP de 0,08 a 0,19).

Como treinar: como as duas cabeças recebem os atributos do tronco desligados (R5), elas são retreinadas sobre o tronco congelado de cada modelo final da L2-12+auto1 (mesma dobra e mesma semente), com 30 épocas, AdamW com taxa de 1 × 10⁻³, lote de 128 e a aumentação da seção 1.12.
F-b1: a função vira classificação em 7 classes pelo prefixo do código (vazio, 03, 003, 103, 24, 55, outro), com perda ponderada pelo inverso da frequência e época escolhida pela revocação média por classe na divisão interna. É comparada com a cabeça de CTC (leitura mapeada para as mesmas classes) pela revocação média por classe, média das sementes, com bootstrap pareado por recorte. É adotada se a diferença tiver o IC acima de zero. Relatam-se o acerto exato, a fração de 103 lidos como 03 ou 003 e se os critérios da seção 1.8 (95% e 2%) são atingidos.
D1: a cabeça de duas leituras passa a usar a média, no tempo, dos atributos do tronco (MLP de 128), com o peso dos positivos igual a negativos ÷ positivos e época escolhida pelo AP interno. O limiar é escolhido pelo F1, com validação aninhada por lote. É adotada se o AP subir com o IC acima de zero e a revocação no limiar escolhido for de pelo menos 0,5.
Reserva (D2): se a D1 não for adotada, na ET6c o digital fora da faixa alta é relido pela faixa de cima, e a confirmação falsa dessa segunda tentativa é medida.
Em qualquer caso, o validador 13.3 (ET6c) só aceita a função com confiança mínima.

R12 (23/11/2026, antes do retreino das cabeças). A F-b1 foi adotada pela regra da R11 (revocação média por classe de 0,285 para 0,402), mas 54% dos 103 continuam lidos como consumo, e em 11 de 98 displays com função que não é de consumo (IC 95% de 5,7% a 19,2%) o valor mostrado é igual ao do CSV. Nesses casos, só a função barra a aprovação.

R13 (23/11/2026, durante o julgamento da P8). A lista de causas da seção 1.13 ganha:

"segmento fantasma": segmento desligado de um display LCD que aparece levemente visível e é lido como aceso;
"tambor em transição": tambor girando de verdade entre dois números; a transição aparente, pelo ângulo da foto, conta como inclinação ou perspectiva;
"linha errada": display de duas leituras em que o leitor leu a de baixo.

Motivo: casos frequentes encontrados no julgamento. Os pares mais confundidos nos digitais diferem por um único segmento (5 e 9, 6 e 8, 9 e 8, 3 e 9); o último tambor responde por metade dos erros de um dígito nos ciclométricos; e a cabeça de duas leituras encontra 56% dos displays de duas linhas. O gatilho dos cantos não muda.

Rotulagem: a função dos 1.481 digitais da fila F-2 ainda sem julgamento recebe rotulagem humana às cegas: sem a função do professor e sem o CSV. É anotado o código como aparece, "vazio" ou "ilegível". Na dúvida entre dois códigos, ilegível. Só a função é julgada; o treino do texto não muda.
Uso: os códigos rotulados entram como itens só de função (o recorte inteiro, ou a faixa de cima quando o professor marcou duas leituras), no treino e na avaliação fora da dobra. As cabeças são retreinadas como na R11. A cabeça nova e a F-b1 são comparadas no conjunto ampliado, as duas com predições refeitas, pela regra da R11 (revocação média por classe). Relatam-se o acerto com 03 = 003, a fração de 103 lidos como consumo e a fração de funções que não são de consumo lidas como consumo, por código.
Motivo: só 67 exemplos de 103 no total (variância por falta de dados, como nos ciclométricos da R9).

R14 (23/11/2026, antes da ampliação da amostra L-a). Com o comitê adotado, o limiar inferior recalculado (R10, a) ficou em log −1,585, com 5 "sim" em 55 recortes abaixo (9,1%): um "sim" a mais o invalidaria. A amostra de legibilidade é ampliada de 300 para 600 recortes: os 300 seguintes da mesma ordem sorteada, julgados pelos mesmos critérios e sem ver as confianças. O limiar é recalculado uma vez, com a confiança do comitê, sobre os 600, sem nova ampliação. Ampliar depois de ver o resultado é uma decisão condicionada ao dado, declarada aqui.

R15 (24/11/2026, antes de qualquer medida da regra da função; ET6c). Regra de aceitação da função (validador 13.3) e decisões da tabela de decisão do leitor.

(a) Consumo: P(vazio) + P(03) + P(003) da cabeça da função (V-a: o display sem código conta como consumo, com a coluna Funcao em NAN). A função é aceita quando essa soma é de pelo menos o limiar t.
(b) Medida, fora da dobra: comitê das 3 sementes de cada dobra, com as probabilidades da cabeça da função (R12) e as do texto pela média das 3 redes. A leitura segue o fluxo do produto: recorte inteiro e, se a D1 disparar (limiar da R11), a faixa de cima. População: os recortes digitais com rótulo de função (aceitos, F-2 e R12). Um recorte é aprovado em t quando a leitura livre é igual ao CSV (comparação numérica) e o consumo é de pelo menos t. Erro: aprovado com função rotulada fora de {vazio, 03, 003}.
(c) Ponto cego (Q-a): antes da medida, 300 aceitos digitais, sorteados com a semente em texto "r15-aceitos", têm a função julgada às cegas, como na R12 (sem a função do professor e sem o CSV). Nesses 300, vale o julgamento. Se houver k > 0 funções que não são consumo entre os julgados, os aceitos não julgados que forem aprovados somam, ao número de erros, a fração k/300 deles, arredondada para cima.
(d) Regra (Y-f2): o menor t, numa grade com os 201 quantis do consumo, cujo limite superior unilateral de 95% (Clopper-Pearson) da fração de erros entre os aprovados seja de no máximo 0,7% (a meta da E35). Relata-se a fração dos consumos mantidos (consumo de pelo menos t) e a dos aprovados.
(e) Gatilho G-a: se, no t escolhido, menos de 50% dos consumos forem mantidos, roda a F-x4: um classificador só da função, com o início da MobileNetV3-Small até a camada 12 (ImageNet, ajuste fino completo) e a cabeça da F-b1, na faixa esquerda de 40% da largura do recorte, redimensionada para 48 × 96; 3 dobras × 3 sementes, 30 épocas, época escolhida pela revocação média interna. Adotada pela regra da R11; se adotada, a regra (d) é refeita com ela.
(f) Tambor em transição (B10-b, decisão de produto): no ciclométrico, uma leitura livre que difere do CSV em exatamente 1 confirma o CSV. Só relato: cobertura dos ciclométricos com e sem a tolerância, e casos reais de CSV diferente do valor verdadeiro em 1 (R9). Um impostor de ±1 não é medido: pela regra, ele é confirmado sempre que o valor verdadeiro é lido.
(g) Tabela de decisão (sem medida): faixa muito baixa vai para "Falha na extração", salvo número de série igual ao esperado, que vai para "Verificar" com "Confiança baixa" (M-b); sem número de série, motivo "Número do medidor não localizado" (S-c); leitura diferente do CSV, motivo "Leitura de consumo divergente do registro", com o valor lido; G1 definitivo; sem a cláusula da orientação; classe "outro" com Funcao em NAN.

Limitações: o limiar é escolhido e certificado nas mesmas predições (como na E30), o que torna a medida otimista; a população rotulada concentra outras funções (fila F-2), o que a torna pessimista. O juiz sem viés é a A1-b no 03/07.

R15-b (24/11/2026, depois do julgamento Q-a e antes de qualquer medida da curva da função). Julgamento dos 300 aceitos (amostra SHA-256 `90462ae4…7b09`): 253 consumo (03, 003 e um combinado 0312), 43 vazio, 2 de função 24, 1 ilegível, 1 que não é display digital. Ponto cego confirmado: 2 em 298 com função julgada (0,67%; limite superior de 95% de cerca de 2,1%).

(a) Substitui o ajuste do item (c) da R15, que somava a fração fixa k/300 aos não julgados aprovados e não dava crédito à cabeça da função. Estimador de razão: em cada limiar t, sejam m(t) os aceitos julgados aprovados em t e k(t) os de função que não é consumo entre eles; aos u(t) aceitos não julgados aprovados em t somam-se ⌈u(t) × k(t)/m(t)⌉ erros. Se m(t) = 0, usa-se a fração geral, 2/298. Os 2 julgados sem função (ilegível e não digital) saem da medida. Revisão feita depois de ver o resultado do Q-a e antes da curva; declarada como tal.
(b) Se nenhum limiar cumprir a regra do item (d) da R15, a medida conta como 0% dos consumos mantidos e o gatilho G-a dispara (F-x4). Se, com a F-x4, ainda nenhum limiar cumprir, os digitais vão para "Verificar" com "Função incorreta", e a regra da função fica como pendência a negociar com o cliente.



### 1.15 Limitações declaradas

- Os alvos privilegiam os recortes fáceis: só entra o que o professor leu inteiro
  e que concorda com o CSV. A cobertura e a confirmação falsa valem para esses
  recortes; o risco real é medido na A1-b (ET6c).
- O impostor é um modelo dos erros do CSV; erros reais de digitação não são
  necessariamente parecidos com o que a tela mostra.
- O limiar alto é escolhido e medido nas mesmas predições (otimista, como na
  E30).
- O julgamento da amostra de legibilidade e das causas dos erros é feito por uma
  pessoa só.
- A faixa de 0 a 55% da altura nos displays de duas leituras é uma regra fixa,
  sem medição prévia.
- Os pesos ImageNet do L2 não declaram licença (mesma situação das E28 e E85).

## 2. Resultados (ET6b, 20 a 23/11/2026)

Predições fora da dobra (3 lotes de desenvolvimento x 3 sementes); o lote 03/07 não
foi usado. Números com 3 sementes são médias, salvo indicação.

### 2.1 Candidatos (seção 1.9; rótulos da ET6a)

| Candidato | M-a interna | M-c fora da dobra | Latência p95 (CPU, 4 threads) |
| --- | --- | --- | --- |
| L1 | 0,244 (0,013 a 0,558) | 0,231 | 9,0 ms |
| L2-9 | 0,672 | 0,607 | 9,0 ms |
| L2-12 | 0,706 | 0,638 | 11,5 ms |

L2-9 contra L2-12: diferença de M-c de -0,031, IC 95% [-0,043; -0,019]; não é
não inferior (Δ = 3 pontos). Vencedor: L2-12. Sem alarme de subtreino.

### 2.2 Diagnóstico dos ciclométricos e rotulagem R9

Acerto exato do L2-12 nos ciclométricos: 0,919 nos vistos, 0,618 na divisão interna
e 0,474 fora da dobra (variância e mudança entre lotes). Erros: 14% no último tambor,
14% com dígito a mais ou a menos. Rotulagem R9 dos 868 recortes sem alvo: 278 alvos
novos, 463 ilegíveis, 115 que não eram ciclométricos, 12 obstruídos. Dos 1.216
recortes ciclométricos do desenvolvimento, 51% são legíveis. Mantidos contra o CSV:
44, dos quais 15 com a mesma leitura do Haiku.

### 2.3 Autotreinamento e ablações (seções 1.9 e 1.10; R8 e R9)

| Fase | Diferença de M-c | IC 95% | Decisão |
| --- | --- | --- | --- |
| Autotreinamento, rodada 1 (cerca de 900 pseudo-rótulos por dobra) | +0,040 (0,633 para 0,673) | [+0,032; +0,049] | Adotado |
| Gatilho da S-b (ciclométricos) | +0,029 | [+0,009; +0,051] | Não dispara |
| Cinza | -0,015 | [-0,023; -0,006] | Não adotado |
| Sem `sonnet_csv` | -0,010 | [-0,018; -0,002] | Não adotado |
| Leitor por família (R8, ajuste fino) | -0,004 | [-0,011; +0,003] | Não adotado; época 0 em 7 de 9 treinos |

Configuração adotada: L2-12+auto1. Limiar alto de -inf em todas as classes e
sementes (a igualdade com o CSV já basta); limite superior da confirmação falsa dos
impostores de 0,041 a 0,045 nos ciclométricos, de 0,025 a 0,027 nos digitais e de
0,009 a 0,010 nos números de série. Cobertura: 0,471 nos ciclométricos, 0,743 nos
digitais e 0,805 nos números de série.

### 2.4 Comitê das 3 sementes (R10, b)

M-c de 0,710 contra 0,673 do modelo único: diferença de +0,037, IC 95% [+0,030;
+0,043]. Latência p95 dos 3 modelos de 29,5 ms (máquina em uso normal). Adotado.

### 2.5 Confiança (seção 1.6)

Temperatura ajustada pela entropia cruzada entre a confiança da leitura livre e o
acerto (nenhuma decisão depende dela). Texto: T = 1,79; ECE de 0,165 a 0,187 para
0,031 a 0,071 no lote fora do ajuste. Função (cabeça de CTC): T = 1,60; ECE de 0,102
a 0,135 para 0,023 a 0,071. A calibração do comitê não foi medida.

### 2.6 Faixas e limiar inferior (seção 1.11; R10 e R14)

Amostra L-a de 600 recortes de display (R14): 339 sim, 79 outro valor, 182 não dá
para ler. Limiar inferior do comitê: log -1,599 (probabilidade 0,202), com 11 "sim"
em 116 recortes abaixo (9,5%); com 300 recortes, tinha sido log -1,585. Na faixa
alta: 197 recortes, nenhum "outro valor". Cobertura da faixa alta: 33% dos displays
e 57,5% dos confirmáveis (60% nos digitais, 44% nos ciclométricos). Faixa muito
baixa: 19% dos displays.

### 2.7 Função e duas leituras (seção 1.8; R11 e R12)

- Cabeça de CTC: acerto de 0,806; 66% dos 103 lidos como 03 ou 003 (F-b dispara).
- F-b1 (classificação sobre o tronco congelado): revocação média por classe de 0,285
  para 0,402, IC [+0,075; +0,162]; adotada.
- D1 (duas leituras): AP de 0,137 para 0,618, IC [+0,424; +0,531]; revocação de 0,563
  no limiar do F1 (precisão de 0,70 a 0,77); adotada.
- Risco real (F-x0): em 11 de 98 displays com função que não é de consumo, o valor
  mostrado é igual ao CSV (IC 95% [0,057; 0,192]); no 24, em 7 de 14.
- R12: 1.481 recortes da F-2 julgados (1.067 com função, 167 vazios, 245 ilegíveis);
  150 exemplos novos de 103. Revocação média por classe de 0,418 para 0,532, IC
  [+0,078; +0,151]; adotada. Lidos como consumo: 103, de 56% para 35%; 24, de 47%
  para 27%; 55, de 44% para 30%.
- Limiar de P(03) + P(003): para aceitar no máximo 2% das outras funções como
  consumo, só 9,7% dos consumos passam (3,4% pelo limite superior). A função não
  atinge os critérios da seção 1.8.

### 2.8 Casos reais de CSV diferente do valor verdadeiro

Ciclométricos (R9): 0 confirmações falsas em 117 leituras. Digitais (F-2): 13 em 324
(5 recortes); no julgamento, em 2 o CSV estava certo, em 2 ficou indefinido e em 1 o
rótulo estava certo.

### 2.9 Análise de erros (seção 1.13; R13)

50 erros (20 digitais, 16 números de série, 14 ciclométricos): outro 23, borrão 11,
reflexo 5, segmento fantasma 5 (todos digitais), campo cortado 2, inclinação ou
perspectiva 1, oclusão 1, transição 1, linha errada 1, rótulo errado 0. Gatilho dos
cantos: 2%, não dispara. Campo cortado: 4%, abaixo de 10%. Função 103 lida como
consumo: o "1" nítido em 8 de 10.

### 2.10 Limitações acrescentadas

- Várias decisões (R9, R11, R12, R14) foram informadas por resultados fora da dobra;
  as comparações seguintes ficam um pouco otimistas. O juiz sem viés é o lote 03/07.
- Rotulagens R9, R12, L-a e P8 feitas por uma única pessoa.
- A cobertura sobre os rótulos (M-c) é maior que a da população (amostra L-a).

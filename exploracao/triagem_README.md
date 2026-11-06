# Triagem de cinco classes (exploração do produto)

Estudo que decide a triagem do produto (spec v11, 2.5, itens 5 a 8; E49 a E53).
Não é entrega de rubrica (instruções, 6.2).

## 1. Protocolo de avaliação

Fixado em 31/10/2026, antes de qualquer treino dos candidatos. Mudanças só por
revisão datada (seção 1.11); o horário do commit é a prova.

### 1.1 Dados e rótulos

- Rótulos de desenvolvimento: `rotulos_grupos.csv`, na pasta de trabalho
  (SHA-256 `e521acd484c3b46d3b5cd312edc327dbc3e830917a5d1185030ff2384cd3d5b4`),
  gerado por `training.build_group_labels` a partir da rotulagem completa
  (9.301 fotos dos lotes 20/05, 21/05 e 22/05; E48). A classe do produto sai
  do grupo por `training.group_labels`, a fonte única.
- Fusão dos grupos com menos de 30 fotos, pela descrição (opção U-b):
  Digital 5 - Pequenino → Digital 1 - Quadrado - Branco; Outros - Poste
  somente → Outros - Sem contexto. Ficam 12 grupos; a fusão nunca cruza a
  classe do produto.
- Treino: fotos íntegras, pareadas e órfãs. Avaliação: só as pareadas, porque
  em produção as órfãs são rejeitadas antes da triagem (E11).
- Sem enriquecimento: os lotes foram rotulados por inteiro, então a
  prevalência de desenvolvimento é a natural, e não há correção de prevalência.
- Teste (03/07): rotulado por inteiro (seção 1.9), fora de qualquer treino,
  calibração ou escolha de limiar.

### 1.2 Partição

- Validação por lote deixado de fora (opção Q1-a): 3 dobras; cada lote é
  avaliado por um modelo treinado nos outros dois; predições fora da dobra
  reunidas.
- 3 sementes por candidato e dobra, combinadas pela média dos logits (E31).
Revisado em 1.11 (R1): uma execução por dobra no C0 e no C1.
- Divisão interna para as épocas (opção E-b), só nos candidatos com ajuste
  fino: em cada dobra, 10% das fotos pareadas dos lotes de treino,
  estratificados por grupo, com semente fixa, escolhem a época pela perda
  logarítmica das classes do produto. O lote avaliado nunca participa. O
  modelo final, treinado com os 3 lotes, usa a mediana das épocas escolhidas
  nas dobras.

### 1.3 Candidatos

Ordem de simplicidade (opção S-a, custo de inferência em CPU no piso da E04):
C0, C2, C4, C1, C3.

| Candidato | Entrada | Treino |
| --- | --- | --- |
| C0 | MobileNetV3-Large congelada (ImageNet V2), 224 px no lado menor, realce fixo, as mesmas 6 vistas do C1 | Sonda linear: regressão logística multinomial (scikit-learn, lbfgs, C = 1,0, até 2.000 iterações), características padronizadas pelo treino de cada dobra |
| C1 | DINOv2 ViT-S/14 congelado, cache da ET4a (6 vistas no treino, a original na avaliação) | A mesma sonda linear do C0 |
| C2 | MobileNetV3-Large (ImageNet V2), rede inteira, resolução nativa (480 × 360) | AdamW, lr 3 × 10⁻⁴, decaimento de pesos 0,05, até 15 épocas, cosseno com 1 época de aquecimento, lote 32 |
| C3 | DINOv2 ViT-S/14, rede inteira, 224 px no lado menor com corte em múltiplo de 14 (E37) | AdamW, lr 2 × 10⁻⁵ na espinha e 1 × 10⁻³ na cabeça, decaimento 0,05, até 10 épocas, cosseno, lote 32 |
| C4 | Igual à C2 (aluno) | Perda = 0,5 · entropia cruzada + 0,5 · T² · KL(professor ‖ aluno), T = 4; professor = C3 da mesma dobra e semente |

Revisado em 1.11 (R2): as vistas do C0 e do C1 vêm de um cache novo, sem
recorte; o cache da ET4a não é usado pela triagem.

Regras comuns:

- Saída: 12 logits de grupo; as probabilidades são somadas nas 5 classes do
  produto (`PRODUCT_CLASSES`).
- Pré-processamento do produto (`preprocess.py`, com o realce fixo) em todos.
- Aumentação (C2, C3 e C4): rotação de ±10°, brilho e contraste de ±0,2,
  translação de até ±5%. Sem recorte aleatório (cortaria o segundo medidor
  dos múltiplos ou parte do medidor) e sem espelhamento.
- Precisão mista (AMP) permitida no treino; toda predição avaliada ou
  entregue é calculada em FP32, sem TF32 (E33).
- Sem memória para o lote 32: acumulação de gradiente (2 passos de 16).
- O C4 só é treinado se nem o C0 nem o C2 forem não inferiores ao melhor
  entre C0 e C3 (seção 1.7).

### 1.4 Perda e pesos de classe

- Padrão: entropia cruzada sem pesos nos 12 grupos (opção W-b).
- Ablação W-d: cada foto recebe o peso da sua classe do produto,
  N / (5 · n_classe). Feita no C0 (família congelada: C0 e C1) e no C2
  (família ajustada: C2, C3 e C4). A W-d só é adotada numa família se tiver
  cobertura maior, com o IC 95% da diferença excluindo zero.
- Com a W-d, os logits são corrigidos (logit − log w) antes da temperatura (E31).

### 1.5 Rótulo fino × amplo

- Padrão: treino nos grupos (fino), conforme a spec.
- Ablação no C0: treino direto nas 5 classes (amplo). O fino só é trocado se
  o amplo tiver cobertura maior, com o IC 95% da diferença excluindo zero
  (opção G-a).

### 1.6 Decisão de rejeição, calibração e limiar

- Escore único (opção R-a): P(rejeitável) = P(outros) + P(indeterminado) +
  P(múltiplos). Rejeita se o escore ≥ limiar; o destino é a maior das três.
- Calibração (opção K-a): uma temperatura por candidato, aplicada aos logits
  de grupo antes da soma e ajustada pela perda logarítmica das classes do
  produto, sem pesos. Validação aninhada por lote: as predições de cada lote
  são calibradas com a temperatura ajustada nos outros dois. A temperatura de
  implantação é ajustada nos 3 lotes.
- O limiar é escolhido depois da calibração, sobre as predições calibradas
  fora da dobra das fotos pareadas dos 3 lotes.

### 1.7 Métricas, meta e regra de escolha

- Perda: medidores determinados (digital ou ciclométrico) pareados rejeitados
  ÷ medidores determinados pareados. Revocação de medidores determinados =
  1 − perda.
- Cobertura da rejeição: rejeitáveis (outros, indeterminado e múltiplos)
  pareados rejeitados ÷ rejeitáveis pareados.
- Meta (opção L2): perda ≤ 0,5%, verificada pelo limite superior unilateral
  de 95% de Clopper-Pearson. Entre os limiares que cumprem a meta, vence o de
  maior cobertura. Sem meta mínima de cobertura.
- Comparação (opção M-b): cada candidato no seu limiar da meta; compara-se a
  cobertura.
- Regra de escolha (opção Δ2): percorre-se a ordem da seção 1.3; vence o
  primeiro candidato cujo limite inferior do IC 95% da diferença (cobertura
  dele − cobertura do melhor) seja maior que −0,05. IC por bootstrap pareado
  sobre as fotos (2.000 reamostragens, semente 0), com os limiares fixos.
- Revisado em 1.11 (R3): a regra de escolha é aplicada depois da revisão das
  discordâncias, com os rótulos corrigidos.

### 1.8 Relatórios obrigatórios

- Perda e cobertura por lote e por condição (02_dataset, 4.3).
- Destino trocado entre as três rejeições.
- Matriz de confusão das classes do produto e a separação entre
  "Ciclométrico - Duas leituras" e múltiplos (E52).
- ECE em 10 faixas para P(rejeitável) e para cada classe de rejeição, antes e
  depois da temperatura.
- Resultados das ablações W-b × W-d e fino × amplo.

### 1.9 Teste 03/07

- Rotulagem completa das fotos íntegras e pareadas, antes de qualquer treino
  dos candidatos, com os mesmos grupos, condições e critérios do
  desenvolvimento: clusters do DINOv2 guiando a ordem e conferência humana
  foto a foto (opção TC-a). As 75 fotos do Lab02 são rotuladas de novo, sem
  consultar os rótulos antigos.
- Abertura única, com o vencedor, a temperatura e o limiar congelados. É a
  segunda abertura do 03/07 (a primeira foi a da E44).
- Meta verificada no teste: perda ≤ 0,5% pelo limite superior de
  Clopper-Pearson. Relatórios da seção 1.8, no teste.

### 1.10 Limitações declaradas

- Um único rotulador no desenvolvimento e no teste, sem concordância entre
  pessoas. A concordância com os rótulos do Lab02, do mesmo rotulador, foi de
  239 em 260 (kappa de 0,811 nas 5 classes e de 0,829 em medidor × outros).
- Os rótulos do teste foram guiados pelos clusters do DINOv2, o extrator do
  C1 e do C3: a independência entre os rótulos do teste e esses candidatos
  não é total.
- O autor viu 75 fotos do teste na ET2 e conhece os números agregados da E44.

### 1.11 Revisões

Revisões de 05/11/2026, feitas antes de qualquer treino dos candidatos.

**R1 — Sementes nas sondas lineares (seção 1.2).** O C0 e o C1 têm uma
execução por dobra. A regressão logística com lbfgs é determinística (o
scikit-learn ignora `random_state` nesse solver), e as vistas são fixas por
pseudônimo; três sementes produziriam modelos idênticos. As 3 sementes valem
para o C2, o C3 e o C4. Evidência no notebook: coeficientes idênticos com
dois valores de `random_state`.

**R2 — Aumentação única, sem recorte (seção 1.3).** As vistas do cache da
ET4a têm recorte aleatório de 80% a 100% da área, que poderia cortar o
segundo medidor dos múltiplos, o mesmo motivo que exclui o recorte do C2 ao
C4 (E57). Passa a existir uma única função de aumentação da triagem, no
pacote `training`: rotação de ±10°, brilho e contraste de ±0,2 e translação
de até ±5%, com os cantos preenchidos pela cor média da foto, aplicada depois
do realce. Ela gera as 5 vistas aumentadas do C0 e do C1 (semente por
pseudônimo e vista), num cache novo com nome próprio, e faz a aumentação
durante o treino do C2, do C3 e do C4. O cache da ET4a fica intacto, para o
artigo de ML.

**R3 — Revisão das discordâncias e correção dos rótulos (seção 1.7; E54).**

- Momento: depois de treinar o C0, o C1, o C2 e o C3, e antes da regra Δ2.
- Fotos revisadas: a união das 50 maiores discordâncias de cada um desses
  candidatos, só entre as pareadas do desenvolvimento, ordenadas pela menor
  probabilidade calibrada (temperatura aninhada) da classe do produto do
  rótulo.
- Revisão às cegas: o autor escolhe o grupo e a condição vendo só a foto,
  sem o rótulo atual nem a predição; só quando a escolha difere do rótulo
  atual os dois aparecem para a decisão final. Ferramenta em `local/`.
- As correções geram uma nova versão do arquivo de rótulos, com novo SHA-256;
  a versão original é preservada.
- As métricas de todos os candidatos são recalculadas com os rótulos
  corrigidos, a partir dos logits fora da dobra gravados; a temperatura, o
  limiar e a regra Δ2 usam os rótulos corrigidos. Os modelos das dobras não
  são retreinados. O modelo final e o C4, se for necessário, treinam com os
  rótulos corrigidos.
- Gatilho de retreino: se as correções passarem de 1% das fotos de
  desenvolvimento, ou de 10% de alguma classe de rejeição (indeterminado,
  múltiplos ou outros), todos os candidatos são retreinados com os rótulos
  corrigidos.
- Os rótulos do teste (03/07) nunca são corrigidos a partir de predições de
  modelo.

Revisões de 06/11/2026, feitas depois da regra Δ2 e antes da abertura do teste.

**R4 — Papéis na abertura do teste (seção 1.9).** A regra Δ2 escolheu o C3; o
C4 falhou por 0,26 ponto (limite inferior de −5,26 contra −5,00). O produto
adota o C4 por decisão de produto, fora da regra: custo de inferência estimado
entre 2 e 5 vezes menor, a medir na ET4g, e diferença de qualidade dentro do
ruído de treino medido (o C3 retreinado como professor deu 64,0% de cobertura,
contra 67,8% na execução avaliada e 65,3% do C4). A abertura avalia os dois,
com os papéis fixados aqui:
C4, o modelo do produto, com a meta verificada; C3, o vencedor da regra do
estudo, relatado à parte. Nenhum dos dois é escolhido pelo resultado do teste.
Ressalva registrada: no desenvolvimento, o C3 aponta mais múltiplos (39 de 72,
contra 27 de 72 no C4).

**R5 — Modelo final (seções 1.2 e 1.6).** Cada modelo final é treinado com as
fotos dos 3 lotes (pareadas e órfãs), com 3 sementes combinadas pela média dos
logits, e o cronograma completo do candidato (cosseno de 15 épocas no C4 e de
10 no C3), parando na mediana das épocas escolhidas na execução R3 do
candidato, sem divisão interna. Os professores do C4 final são os C3 finais da
mesma semente. A temperatura e o limiar são os da execução R3 (temperatura
ajustada nos 3 lotes e limiar da meta sobre as predições calibradas fora da
dobra), congelados antes da abertura.

**R6 — Artefato entregue, latência e ordem da abertura (seções 1.6 e 1.9).**
Revisão de 06/11/2026, feita antes do treino dos modelos finais.

- Motor: o artefato da triagem do produto é um módulo PyTorch com as 3 redes
  do C4 final, a temperatura e o limiar (E32). O motor definitivo continua
  sendo decidido na ET9 (E34). O ONNX é exportado e medido como evidência:
  paridade dos logits e decisões idênticas nas fotos pareadas do
  desenvolvimento. Comparar as decisões dos dois motores nas fotos do teste
  não lê rótulos e não conta como abertura.
- Saída: o artefato devolve as 12 saídas de grupo, com a média dos logits das
  3 sementes e a temperatura já aplicadas. A soma por classe usa o mapeamento
  grupo → classe, que passa para o pacote do produto; o pacote `training`
  passa a importá-lo de lá. O escore de rejeição e o destino seguem a seção
  1.6.
- Modelo final: épocas de parada iguais à mediana das 9 épocas escolhidas
  (3 lotes × 3 sementes) na execução R3 de cada candidato, lidas dos
  metadados gravados (`c3_wb_fino_r3` e `c4_wb_fino_r3`). Pesos e metadados
  (commit, SHA-256 dos rótulos e dos pesos, épocas, temperatura e limiar)
  ficam na pasta de trabalho, em `modelos/`.
- Gatilho de latência: medida antes da abertura, em CPU com 4 threads, lote
  de 1, com pré-processamento, p50 e p95 em 200 fotos pareadas do
  desenvolvimento, depois de 20 fotos de aquecimento, no PyTorch (motor do
  produto) e no ONNX Runtime, para o C4 e para o C3 (3 redes cada). Se o p50
  do C3 no PyTorch for menos de 2 vezes o do C4, a abertura é suspensa e a
  E65 é revista, com nova revisão deste protocolo antes da abertura.
- Ordem: (1) treino dos modelos finais; (2) artefato e paridade ONNX;
  (3) latência e gatilho; (4) ensaio do código da abertura sobre os logits
  fora da dobra da execução R3, que precisa reproduzir os números da E64;
  (5) `triage.py` do produto, conferido contra a predição do treino nas fotos
  de desenvolvimento; (6) abertura única do teste a partir das fotos em
  disco, com o C4 pelo `triage.py` do produto (meta verificada) e o C3 pelo
  código de treino, com o mesmo pré-processamento (relatado), com os
  relatórios da seção 1.8. Os logits do teste e o resultado são gravados uma
  vez e nunca recalculados.
- Local: blocos novos em `triagem_candidatos.ipynb`, a partir do bloco 16.

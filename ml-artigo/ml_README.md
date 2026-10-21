# Triagem medidor × outros sem rótulos (rubrica de ML)

## 1. Protocolo de avaliação

Fixado antes de qualquer extração de características ou agrupamento.
Mudanças só por revisão datada (seção 1.10); o horário do commit é a prova.
Os rótulos, a partição e o lote de teste são os do Lab02 (E11). A triagem
de ML é comparada e reportada; a triagem do produto continua sendo a de DL.

### 1.1 Dados

- Agrupamento: todas as fotos íntegras dos três lotes de desenvolvimento
  (20/05, 21/05 e 22/05), órfãs incluídas, sem nenhum rótulo.
- Avaliação no desenvolvimento: a amostra aleatória rotulada desses lotes
  (225 fotos, 17 de "outros"). Os 36 zeros ficam fora das métricas.
  A avaliação é transdutiva: as fotos avaliadas participaram do agrupamento.
- Teste (03/07): lacrado; fora do agrupamento. As fotos do teste só recebem
  o cluster pela cabeça já treinada, no momento da abertura (seção 1.9).

### 1.2 Características

- Proposta: DINOv2 ViT-S/14, congelado, pesos oficiais carregados pelo
  `timm` (`vit_small_patch14_dinov2.lvd142m`); característica = token de
  classe (384 dimensões). Pré-processamento igual ao da triagem de DL
  (spec 2.5.7): realce fixo, 224 px no lado menor, normalização do
  ImageNet; cada lado é então cortado ao centro para o maior múltiplo de 14
  (224 × 294 nas fotos de 360 × 480). Fixado a priori, sem comparar
  resoluções pelos rótulos. Se a análise de erros mostrar que os medidores
  mapeados como "outros" são sobretudo pequenos ou distantes, a resolução
  quase nativa (350 × 476) entra só como análise exploratória.
- Comparação de extratores (H2): o mesmo DINOv2, a MobileNetV3-Large a
  224 px (a espinha da triagem de DL, no lugar da ResNet-50 prevista) e a
  baseline manual do Lab02 (HOG, cor e qualidade).

### 1.3 Métodos

- SCAN: cabeça linear sobre o extrator congelado, perda SCAN, depois
  auto-rotulagem.
- Linhas de base: K-means com K = 2 e com K = 50, nas mesmas características.
- Aumentação: 5 vistas aumentadas por foto, extraídas uma vez e guardadas em
  cache; a cada época sorteia-se uma vista. Transformações leves, para não
  tirar o medidor da foto: recorte aleatório de 80% a 100% da área, na
  proporção 3:4; rotação de até 10 graus; brilho e contraste variando até
  20%. Sem espelhamento e sem apagamento de regiões (cutout).
- Perda SCAN: âncora e vizinho em vistas sorteadas. Auto-rotulagem:
  confiança medida na foto original; perda calculada numa vista sorteada.

### 1.4 Hiperparâmetros

Fixos a priori, sem nenhuma escolha por rótulos. Valores da configuração
oficial do SCAN para espinha congelada com só a cabeça linear treinada
(`configs/scan/scan_imagenet_50.yml` e `configs/selflabel/selflabel_imagenet_50.yml`,
`update_cluster_head_only: True`), exceto a aumentação (seção 1.3):

- Agrupamento: K = 50 clusters; k = 50 vizinhos por similaridade de
  cosseno, buscados na foto original; peso da entropia 5,0; SGD com momento
  0,9, lr 5,0, sem decaimento; 100 épocas, lote de 512; 10 cabeças, e fica
  a de menor perda SCAN na última época (critério do artigo, sem rótulos).
  - Entrada da cabeça: vetores normalizados para comprimento 1, a mesma
  geometria dos vizinhos (cosseno) e do K-means.
- Auto-rotulagem: limiar de confiança 0,99, sem balanceamento de classes;
  SGD com momento 0,9, lr 0,03; 25 épocas, lote de 512; média móvel
  exponencial dos pesos (0,999). Só a cabeça é treinada.
- Sanidade, sem rótulos: se a perda divergir (NaN ou crescimento contínuo)
  ou se, ao fim do agrupamento, menos da metade dos clusters estiver ocupada
  (quase-colapso), o lr da fase é dividido por 10 até que isso não ocorra,
  em até 4 tentativas; o valor usado é reportado.

5 sementes por configuração. A sensibilidade a K (20 e 100) é exploratória
e nunca escolhe a configuração.

Geometria, igual em todos os métodos: as características do DINOv2 e da
MobileNetV3-Large são usadas como saem da rede; as da baseline manual são
padronizadas (escore z, com média e desvio das fotos agrupadas, sem rótulos),
porque só ela mistura grandezas de unidades diferentes. Os vizinhos usam a
similaridade de cosseno; o K-means recebe os vetores normalizados para
comprimento 1, em que a distância euclidiana ordena os vizinhos como o cosseno.

### 1.5 Mapeamento dos clusters

- Por maioria, para avaliar: cada cluster recebe a classe da maioria das
  fotos rotuladas dele, contadas nos outros dois lotes (o lote medido nunca
  mapeia a si mesmo). Cluster sem foto rotulada nesses lotes ou empatado:
  "medidor" (o lado seguro, E09). Feito em todas as sementes.
- Humano, para medir o custo (H3): na execução de menor perda SCAN
  (critério do artigo, sem rótulos). O inspetor vê, sem rótulos, as fotos
  de cada cluster mais próximas do protótipo, só dos lotes de
  desenvolvimento; dúvida vai para "medidor". Registram-se o tempo, o
  número de clusters e o de fotos inspecionadas.

### 1.6 Métricas

- Otimizadora: AP de "outros". Escore de "outros" de uma foto =
  soma, sobre os clusters c, de p(c | foto) · π_c, em que π_c é a fração de
  "outros" entre as fotos rotuladas do cluster c nos outros dois lotes,
  suavizada para a prevalência (m = 1). No K-means, p(c | foto) vale 0 ou 1.
- Viabilidade da triagem: revocação de medidor ≥ 0,97 pelo limite inferior
  unilateral de 95% de Clopper-Pearson, no mapeamento rígido, reportada para
  cada configuração. Com 208 medidores, admite no máximo 2 perdidos. Ela não
  é usada sozinha, porque mapear tudo como "medidor" daria 100%.
- Descritivas: revocação de "outros", acurácia balanceada, F1 macro e matriz
  de confusão. Com 17 "outros", nenhuma delas decide nada.
- Variação: média e desvio entre sementes; IC por bootstrap pareado.

### 1.7 Hipóteses e regra de decisão

- H1 (confirmatória): SCAN > K-means com K = 50, no DINOv2, em AP de
  "outros" (qualidade da ordenação, independente do mapeamento rígido).
  Estatística: média, entre as 5 sementes, da AP de cada configuração.
  Vence se o IC 95% da diferença (bootstrap pareado e estratificado por
  classe, 2.000 reamostragens; em cada uma, a AP de cada semente nas mesmas
  fotos, depois a média) excluir 0; senão, empate. A viabilidade da seção
  1.6 é reportada à parte e não entra na decisão da H1.
- H2 (confirmatória): pureza dos 5 vizinhos (cosseno) de cada foto da amostra
  aleatória, buscados entre as próprias 225 fotos, sem a foto, no colapso
  binário. Estatística: pureza balanceada, a média da pureza das fotos de
  medidor e da pureza das fotos de "outros"; as duas são reportadas. DINOv2
  contra cada um dos outros dois extratores: vence se o IC 95% da diferença
  (bootstrap pareado, estratificado por classe, 2.000 reamostragens) excluir 0.
- H3 (descritiva): o custo do mapeamento humano (seção 1.5) e a sua perda
  em relação ao mapeamento por maioria.
- Exploratórias: K-means com K = 2, a sensibilidade a K, o mesmo pipeline
  nos outros extratores e a comparação com a DL.

### 1.8 Comparação com a triagem de DL

No colapso binário (E11): a AP de "outros" da DL fora da dobra, calibrada,
contra a do ML, no desenvolvimento. É apenas descritiva: os dois diferem no
extrator e na supervisão, e a diferença não é atribuível a nenhum dos dois.

### 1.9 Abertura do teste

Uma única vez, depois do mapeamento humano e com tudo congelado:
a triagem de DL (modelo final, limiar 0,966) e a de ML (execução mapeada).
No teste, só a revocação de medidor é estimável (cerca de 74 medidores,
1 foto de "outros"), com IC de Clopper-Pearson. O lado "outros" do teste é
declarado não estimável.

### 1.10 Revisões do protocolo

| Data | Revisão | Motivo |
| 20/10/2026 | Seção 1.2: carregamento pelo `timm` e pré-processamento fixados (224 px no lado menor, corte para múltiplo de 14) | Pendência prevista no próprio protocolo, resolvida antes de qualquer extração ou agrupamento |
| 20/10/2026 | Seções 1.4 e 1.7: estatística única da H2 (pureza balanceada), bootstrap estratificado por classe, padronização só da baseline e vetores normalizados no K-means | O texto original não fixava uma estatística única nem a geometria; fixado antes de qualquer cálculo de pureza ou de agrupamento |
| 20/10/2026 | Seções 1.6 e 1.7: a H1 decide só pela AP de "outros"; a restrição de revocação passa a ser o critério de viabilidade, reportado à parte; bootstrap da H1 estratificado por classe | O texto não definia a H1 quando a linha de base não cumpre a restrição, como ocorreu com o K-means (K = 50) no desenvolvimento; fixado antes de qualquer execução do SCAN |
| 20/10/2026 | Seção 1.4: hiperparâmetros trocados pelos da configuração oficial do SCAN para espinha congelada (ImageNet-50), mais uma regra de sanidade sem rótulos para o lr | Os valores anteriores vinham da configuração do CIFAR-10, em que o SCAN atualiza a rede inteira; o código oficial tem uma configuração própria para o regime deste projeto (só a cabeça linear). Fixado antes de qualquer execução do SCAN |
| 20/10/2026 | Seção 1.4: vetores normalizados antes da cabeça do SCAN; sanidade passa a detectar o quase-colapso (menos da metade dos clusters ocupados). **Revisão posterior a resultados com rótulos.** A H1 é decidida pela execução corrigida; a execução original é reportada | Diagnóstico sem rótulos da primeira execução: norma das características de cerca de 47, logits com desvio de 2.800 a 6.600, softmax saturado e 3 a 6 de 50 clusters ocupados já ao fim do agrupamento. O lr oficial (5,0) foi ajustado para outra escala de características. A correção usa a geometria já fixada para os vizinhos e o K-means; o critério não usa rótulos |

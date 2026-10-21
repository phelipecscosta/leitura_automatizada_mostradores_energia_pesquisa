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

Fixos a priori, sem nenhuma escolha por rótulos: K = 50 clusters;
k = 20 vizinhos por similaridade de cosseno; peso da entropia 5,0; Adam com
lr 1e-4 e decaimento 1e-4; 50 épocas, lote de 128; auto-rotulagem com limiar
0,99 e balanceamento de classes. Valores do código oficial do SCAN, exceto
a aumentação (seção 1.3) e a espinha congelada. 5 sementes por configuração.
A sensibilidade a K (20 e 100) é exploratória e nunca escolhe a configuração.

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
- Restrição: revocação de medidor ≥ 0,97 pelo limite inferior unilateral
  de 95% de Clopper-Pearson, no mapeamento rígido. Ela não é usada sozinha,
  porque mapear tudo como "medidor" daria 100%.
- Descritivas: revocação de "outros", acurácia balanceada, F1 macro e matriz
  de confusão. Com 17 "outros", nenhuma delas decide nada.
- Variação: média e desvio entre sementes; IC por bootstrap pareado.

### 1.7 Hipóteses e regra de decisão

- H1 (confirmatória): SCAN > K-means com K = 50, no DINOv2, em AP de
  "outros". Vence se o IC 95% da diferença (bootstrap pareado, médias das
  sementes) excluir 0; senão, empate. As duas configurações precisam
  cumprir a restrição da seção 1.6.
- H2 (confirmatória): pureza dos 5 vizinhos dentro da amostra rotulada,
  calculada por classe; DINOv2 contra cada um dos outros dois extratores,
  com a mesma regra.
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

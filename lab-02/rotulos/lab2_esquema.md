# Esquema de rótulos — Lab02

Versão: 5 (após a rotulagem completa).


## 1. Princípio

O rótulo descreve o que a foto mostra, e não para onde o software deve
enviá-la. As regras de destino ficam na spec e são avaliadas com estes
rótulos. A preferência de negócio por não perder medidores (E09) entra no
limiar de decisão, nunca no rótulo.

## 2. Procedimento

1. Passada 1, sem o BaseExtracao aberto: `cena_n1`, `cena_n2`,
   `legibilidade`, `incerto`, `campos_visiveis` e `observacao`.
2. Passada 2, com o BaseExtracao aberto, só depois de fechar a passada 1:
   `leitura_confere`.
3. Cada foto é vista em tamanho nativo, com zoom livre, em duas versões: a
   original e a realçada pelo operador fixo
   `PIL.ImageOps.autocontrast(imagem, cutoff=1)`. Nenhuma outra edição é
   permitida.

## 3. Cena, nível 1

| Classe | Definição operacional |
| --- | --- |
| medidor | O rotulador consegue apontar onde está um medidor de energia na foto, em qualquer das duas versões, mesmo sem ler nada |
| outros | Qualquer outro caso |

Regras:
- Vários medidores na foto: medidor.
- Só o display ou só a placa: medidor.
- Medidor visto através da tampa da caixa: medidor.
- Caixa fechada, sem o medidor à vista: outros.
- Hidrômetro, medidor de gás, papel, bilhete ou tela com a leitura anotada: outros.
- Imagem vazia que passou pela integridade: outros, com observação `vazia`.

## 4. Cena, nível 2 (só para medidor)

| Classe | Definição operacional |
| --- | --- |
| digital | Display eletrônico de segmentos ou matricial |
| ciclometrico | Contador de tambores numerados |
| indeterminado | Não é possível distinguir o tipo; só a placa visível; medidor de ponteiros (com observação `ponteiros`) |

Com vários medidores, vale o tipo do medidor principal: o mais legível e,
em caso de empate, o mais central. Se forem de tipos diferentes e não
houver um principal, o tipo é indeterminado.

## 5. Legibilidade (só para medidor)

| Classe | Definição operacional |
| --- | --- |
| ilegivel | Nenhum dígito da leitura nem do número do medidor pode ser lido com segurança, em nenhuma das duas versões |
| legivel | Qualquer outro caso |

Só contam como informação a leitura e o número do medidor; a função
sozinha não conta. Com vários medidores, vale qualquer dígito de qualquer
um deles.

## 6. Colunas

| Coluna | Valores | Regra |
| --- | --- | --- |
| nome_arquivo | código pseudônimo | Nunca o nome real |
| lote | data do lote (AAAA-MM-DD) | |
| rotulador | R1, R2 ou ADJ | ADJ: rótulo final de uma foto rotulada às cegas e revisada em conjunto. O rótulo final de uma foto é o ADJ, se existir; senão, o do R1 |
| cena_n1 | medidor, outros | |
| cena_n2 | digital, ciclometrico, indeterminado | Vazio quando cena_n1 = outros |
| legibilidade | legivel, ilegivel | Vazio quando cena_n1 = outros |
| incerto | sim, nao | sim: o rótulo é o melhor palpite, mas há dúvida real |
| campos_visiveis | leitura, numero, ambos, nenhum | Campos com ao menos um dígito legível. Vazio quando cena_n1 = outros. nenhum se e somente se legibilidade = ilegivel |
| leitura_confere | sim, nao, impossivel | Passada 2. Comparação numérica com a leitura do BaseExtracao (sem zeros à esquerda). impossivel: a leitura de consumo (função 03, nos digitais) não aparece por completo na foto, seja por dígitos ilegíveis, seja por display mostrando outra função (com observação outra_funcao) ou função não identificável. Vazio quando cena_n1 = outros, para o R2 (que não faz a passada 2) e para fotos órfãs (sem linha no BaseExtracao, portanto sem valor de referência) |
| observacao | vazia, multiplos, ponteiros, outra_funcao, outro | Vocabulário fechado. Nunca texto livre. outra_funcao: medidor digital com o display mostrando uma função diferente de 03 (por exemplo, 103, energia injetada na rede) |

## 7. Exemplos-limite

Dois por fronteira, um em cada sentido. Citados pelo código pseudônimo, com
descrição em texto; nunca como imagem.

### Medidor × outros

- **F-0482fcbf7800**. Parece medidor: tons de cinza, traços retos e quadrados
  escuros. É outros: parede de azulejos com um bilhete ao centro; nenhum
  medidor pode ser apontado (seção 3; bilhete é outros).
- **F-64885ea97b45**. Parece outros: fundo claro com uma figura escura no
  centro, que lembra um buraco ou um poço. É medidor: a foto está escura, mas
  a luz de LED sinaliza o aparelho, que pode ser apontado (seção 3).

### Digital × ciclométrico

- **F-d7625174368e**. Parece digital: caixa quadrada, de aspecto moderno.
  É ciclométrico: a leitura está nos tambores numerados, de cor mais escura
  (seção 4).
- **F-95f1257deb1f**. Parece ciclométrico: caixa redonda e visor escuro.
  É digital: o visor é um LCD de fundo escuro, variante menos comum do LCD
  cinza (seção 4).

### Legível × ilegível

- **F-492ed77c7da8**. Parece legível: o mostrador ciclométrico está visível e
  pouco desfocado. É ilegível: os tambores estão perto da virada e a foto foi
  tirada inclinada; nenhum dígito se distingue com segurança (9, 8 ou 0), nem
  com autocontraste, e a placa não tem dígito legível (seção 5). Ilustra
  também a passada 1 sem o BaseExtracao (seção 2): a impressão de que a
  leitura era conhecida veio só do valor registrado.
- **F-6249d58c5d12**. Parece ilegível: dígitos de baixo contraste, de formatos
  semelhantes, difíceis de distinguir na foto original. É legível: com o
  autocontraste fixo, os dígitos da leitura são lidos com segurança (seção 5).

### Indeterminado × tipo conhecido

- **F-eb6e3c96f15c**. Parece digital: caixa retangular, de aspecto moderno.
  É indeterminado: o visor é escuro e tem marcas brancas, que podem ser a
  numeração dos tambores de um ciclométrico ou sujeira e respingos de tinta
  sobre o visor escuro de um digital (há respingos iguais ao redor da
  caixa). Nenhum indício decide o tipo; o autocontraste escurece o visor e
  realça as marcas, mas não as distingue (seção 4).
- **F-1768dabc776b**. Parece indeterminado: nenhum dígito se lê. É
  ciclométrico: o contorno dos tambores, de fundo preto e numeração branca,
  é visível, e o autocontraste o deixa mais nítido. O tipo se decide pela
  forma do mostrador, e não pela leitura (seção 4); a legibilidade é um
  julgamento separado (seção 5).

O mesmo indício, visor escuro com marcas brancas, aparece nos dois
exemplos. A fronteira passa por saber se as marcas formam o padrão dos
tambores.

Na rotulagem completa, todos os 28 indeterminados são também ilegíveis e
sem campo visível: os casos de "só a placa visível" e de medidor de
ponteiros, previstos na seção 4, não ocorreram na amostra.

## 8. Histórico de revisões

| Versão | Data | Mudança | Motivo |
| --- | --- | --- | --- |
| 1 | 05/10/2026 | Versão inicial | — |
| 2 | 08/10/2026 | Exemplos-limite (seção 7); definições inalteradas | Piloto às cegas com 50 fotos e dois rotuladores: kappa de 1,00 em medidor × outros, 0,95 na cena com 4 classes e 0,91 em legibilidade. As 6 discordâncias foram revisadas em conjunto e todas tiveram origem operacional (erro de digitação ou de atenção), e não ambiguidade das definições. Em campos visíveis, que não era objeto do kappa, "leitura" e "número" ficaram com kappa de 0,55 e 0,48, pelo mesmo motivo |
| 3 | 08/10/2026 | Regras das colunas rotulador e leitura_confere (seção 6); definições inalteradas | Rotulador ADJ para as adjudicações do piloto. Fotos órfãs, presentes no treino, não têm leitura de referência: leitura_confere fica vazio, para não entrar na taxa de erro das anotações de campo |
| 4 | 08/10/2026 | Código outra_funcao; definição de impossivel em leitura_confere (seção 6) | Na revisão das 31 leituras divergentes, 22 eram fotos de medidores digitais com o display em outra função (sobretudo 103, de forma parecida com 03). A leitura de consumo não aparecia, então a divergência não era erro de anotação do leiturista |
| 5 | 08/10/2026 | Exemplos-limite do indeterminado (seção 7); definições inalteradas | O piloto não teve nenhum indeterminado; a rotulagem completa teve 28. Exemplos escolhidos fora do lote de teste, por inspeção no visualizador local, sem alterar nenhum rótulo |

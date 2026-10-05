# Esquema de rótulos — Lab02

Versão: 1 (antes do piloto). O histórico de revisões, com o motivo de cada
mudança, fica na última seção.

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
| rotulador | código do rotulador | |
| cena_n1 | medidor, outros | |
| cena_n2 | digital, ciclometrico, indeterminado | Vazio quando cena_n1 = outros |
| legibilidade | legivel, ilegivel | Vazio quando cena_n1 = outros |
| incerto | sim, nao | sim: o rótulo é o melhor palpite, mas há dúvida real |
| campos_visiveis | leitura, numero, ambos, nenhum | Campos com ao menos um dígito legível. Vazio quando cena_n1 = outros. nenhum se e somente se legibilidade = ilegivel |
| leitura_confere | sim, nao, impossivel | Passada 2. Comparação numérica com a leitura do BaseExtracao (sem zeros à esquerda). impossivel: a leitura não é legível por completo. Vazio quando cena_n1 = outros |
| observacao | vazia, multiplos, ponteiros, outro | Vocabulário fechado. Nunca texto livre |

## 7. Exemplos-limite

Dois por classe: a foto que quase é e a que quase não é. Citados pelo
código pseudônimo, com descrição em texto; nunca como imagem. A preencher
no piloto (T2.5).

## 8. Histórico de revisões

| Versão | Data | Mudança | Motivo |
| --- | --- | --- | --- |
| 1 | 05/10/2026 | Versão inicial | — |

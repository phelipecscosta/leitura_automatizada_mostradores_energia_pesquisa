# Leitura automatizada de mostradores de energia: pesquisa

Material acadêmico do projeto de leitura automatizada de fotografias de
medidores de energia de uma concessionária de energia do Nordeste.

O software em si está no repositório do produto:
[leitura_automatizada_mostradores_energia](https://github.com/phelipecscosta/leitura_automatizada_mostradores_energia).
Este repositório contém somente as entregas exigidas pelas disciplinas e os
arquivos criados para elas. Ele usa o código do produto instalado como pacote,
sem copiá-lo.

## Conteúdo

| Pasta | Conteúdo |
| --- | --- |
| `lab-02/` | Entrega do Marco 1 (Lab02) da disciplina de Deep Learning |
| `ml-artigo/` | Artigo da disciplina de Machine Learning |
| `tests/` | Testes de proteção dos dados do cliente |

As pastas que ainda não têm conteúdo contêm apenas um arquivo `.gitkeep`.

## Proteção dos dados do cliente

- Nenhuma foto, planilha de dados, feature, peso ou segredo é versionado.
- Nomes de arquivos e identificadores do cliente aparecem somente como códigos
  pseudônimos.
- Os notebooks versionados não exibem fotos do cliente. A inspeção visual é
  feita em notebooks da pasta `local/`, que é ignorada pelo Git.

## Ambiente

Requer o repositório do produto clonado ao lado deste. No PowerShell, a partir
da pasta deste repositório:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

$produto = "..\leitura_automatizada_mostradores_energia"
python -m pip install -r "$produto\requirements-lock-cu126.txt"
python -m pip install -e "$produto[dev]" --no-deps

Copy-Item .env.example .env   # preencha o caminho dos dados
python -m pytest
```

## Licença

O código está sob a licença MIT (arquivo `LICENSE`). A licença não se aplica a
dados.

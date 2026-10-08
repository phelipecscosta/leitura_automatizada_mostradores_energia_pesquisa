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

# Versões exatas: produto mais as dependências da pesquisa
python -m pip install -r requirements-lock-pesquisa.txt

# O produto, em modo editável, sem buscar outras dependências
python -m pip install -e "..\leitura_automatizada_mostradores_energia[dev,training]" --no-deps


# O pacote de treino do produto (pasta training/) fica fora do pacote
# instalado. Um arquivo .pth no .venv acrescenta a raiz do produto ao caminho
# de importação. Refaça este passo sempre que recriar o .venv.
$sitePackages = python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
$productRoot = (Resolve-Path ..\leitura_automatizada_mostradores_energia).Path
Set-Content (Join-Path $sitePackages "meter_reader_training.pth") $productRoot -Encoding ascii

Copy-Item .env.example .env   # preencha o caminho dos dados
python -m pytest
```

### Notebooks

Os notebooks rodam no VS Code, com o kernel do `.venv` deste repositório.
Notebooks que exibem fotos ou nomes de arquivos do cliente ficam na pasta
`local/`, que é ignorada pelo Git.

### Dependências

- `requirements-pesquisa.txt`: dependências diretas da pesquisa, além das do
  produto.
- `requirements-lock-pesquisa.txt`: versões exatas de todo o ambiente. Contém
  todos os pacotes do travamento do produto, nas mesmas versões.

Para acrescentar uma dependência, inclua-a em `requirements-pesquisa.txt`,
instale usando o travamento do produto como restrição e gere de novo o
travamento da pesquisa:

```powershell
python -m pip install -r requirements-pesquisa.txt `
    -c ..\leitura_automatizada_mostradores_energia\requirements-lock-cu126.txt
python -m pip freeze --all --exclude-editable
```

A saída do `freeze` substitui as linhas de pacotes do travamento, mantendo o
cabeçalho.

## Licença

O código está sob a licença MIT (arquivo `LICENSE`). A licença não se aplica a
dados.

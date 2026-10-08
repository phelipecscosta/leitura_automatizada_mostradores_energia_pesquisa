"""Teste de conteúdo dos rótulos do Lab02 (exceção nominal da decisão E13).

O arquivo é versionado num repositório público. Este teste garante que ele
segue o esquema (lab-02/rotulos/lab2_esquema.md, versão 1) e que nenhum nome
real ou texto livre entra nele. Mensagens indicam a linha, nunca o valor.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

LABELS_PATH = Path(__file__).resolve().parents[1] / "lab-02/rotulos/lab2_rotulos.csv"

COLUMNS = [
    "nome_arquivo", "lote", "rotulador", "cena_n1", "cena_n2", "legibilidade",
    "incerto", "campos_visiveis", "leitura_confere", "observacao",
]
CODE_RE = re.compile(r"F-[0-9a-f]{12}")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
# Valores permitidos; "" = vazio. As colunas de um só rotulador podem ficar vazias.
ALLOWED = {
    "rotulador": {"R1", "R2", "ADJ"},  # ADJ: rótulo final de uma foto revisada em conjunto
    "cena_n1": {"medidor", "outros"},
    "incerto": {"", "sim", "nao"},
    "observacao": {"", "vazia", "multiplos", "ponteiros", "outro"},
}
METER_ONLY = {  # só para medidor; vazias quando cena_n1 = outros
    "cena_n2": {"digital", "ciclometrico", "indeterminado"},
    "legibilidade": {"legivel", "ilegivel"},
    "campos_visiveis": {"", "leitura", "numero", "ambos", "nenhum"},
    "leitura_confere": {"", "sim", "nao", "impossivel"},
}


def validate_labels(text: str) -> list[str]:
    """Lista os problemas do arquivo; lista vazia = arquivo válido."""
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames != COLUMNS:
        return ["Cabeçalho diferente do esquema."]
    errors, seen = [], set()
    for line, row in enumerate(reader, start=2):
        if not CODE_RE.fullmatch(row["nome_arquivo"] or ""):
            errors.append(f"Linha {line}: nome_arquivo não é um pseudônimo.")
        if not DATE_RE.fullmatch(row["lote"] or ""):
            errors.append(f"Linha {line}: lote fora do formato AAAA-MM-DD.")
        for column, values in ALLOWED.items():
            if row[column] not in values:
                errors.append(f"Linha {line}: valor inválido em {column}.")
        is_meter = row["cena_n1"] == "medidor"
        for column, values in METER_ONLY.items():
            valid = row[column] in values if is_meter else row[column] == ""
            if not valid:
                errors.append(f"Linha {line}: valor inválido em {column}.")
        visible = row["campos_visiveis"]
        if is_meter and visible and (visible == "nenhum") != (row["legibilidade"] == "ilegivel"):
            errors.append(f"Linha {line}: campos_visiveis incoerente com legibilidade.")
        key = (row["nome_arquivo"], row["rotulador"])
        if key in seen:
            errors.append(f"Linha {line}: foto repetida para o mesmo rotulador.")
        seen.add(key)
    # Uma adjudicação só existe onde houve dupla rotulagem: exige R1 e R2 da mesma foto
    for code in {c for c, r in seen if r == "ADJ"}:
        if not {(code, "R1"), (code, "R2")} <= seen:
            errors.append("Adjudicação sem as duas rotulagens originais.")
    return errors


HEADER = ",".join(COLUMNS)
VALID = "F-79b380974881,2026-05-20,R1,medidor,digital,legivel,nao,ambos,sim,"


def test_valid_rows_pass():
    other = "F-0123456789ab,2026-05-20,R1,outros,,,nao,,,vazia"
    assert validate_labels(f"{HEADER}\n{VALID}\n{other}\n") == []


@pytest.mark.parametrize("row", [
    VALID.replace("F-79b380974881", "foto_123.jpg"),  # nome real no lugar do código
    VALID[:-1] + ",medidor da esquina",               # texto livre na observação
    "F-0123456789ab,2026-05-20,R1,outros,digital,,nao,,,",  # outros com tipo
    VALID.replace("legivel,nao,ambos", "legivel,nao,nenhum"),  # incoerente
])
def test_invalid_rows_fail(row):
    assert validate_labels(f"{HEADER}\n{row}\n")


def test_wrong_header_and_duplicate_fail():
    assert validate_labels(f"nome_arquivo,lote\n{VALID}\n")
    assert validate_labels(f"{HEADER}\n{VALID}\n{VALID}\n")


def test_versioned_labels_file():
    # Roda sobre o arquivo real quando ele existir (mesmo antes do git add)
    if not LABELS_PATH.exists():
        pytest.skip("lab2_rotulos.csv ainda não existe.")
    assert validate_labels(LABELS_PATH.read_text(encoding="utf-8")) == []

def test_adjudication_with_both_labelers_passes():
    rows = [VALID.replace(",R1,", f",{r},") for r in ("R1", "R2", "ADJ")]
    rows[1] = rows[1].replace(",sim,", ",,")  # R2 não faz a passada 2
    assert validate_labels(HEADER + "\n" + "\n".join(rows) + "\n") == []


def test_adjudication_without_double_labeling_fails():
    adj_only = VALID.replace(",R1,", ",ADJ,")
    assert validate_labels(f"{HEADER}\n{VALID}\n{adj_only}\n")  # falta o R2

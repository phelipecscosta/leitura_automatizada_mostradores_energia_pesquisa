"""Notebooks versionados não podem exibir fotos nem nomes do cliente (instruções 6.2).

Opção 5B da T2.4: gráficos saem em SVG vetorial. Por isso são proibidos, nas
saídas, imagens em bitmap (PNG, JPEG etc.), SVG com bitmap embutido, imagens
em HTML e anexos em células de Markdown. No texto das saídas são proibidos
".jpg"/".jpeg" e sequências de 20 dígitos, que são a forma dos nomes reais.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BITMAP_RE = re.compile(r"<image|<img|data:image", re.IGNORECASE)
CLIENT_TEXT_RE = re.compile(r"\.jpe?g|\d{20}", re.IGNORECASE)


def _as_text(value) -> str:
    """Saídas do notebook guardam texto como string ou lista de strings."""
    return "".join(value) if isinstance(value, list) else str(value)


def check_notebook(notebook: dict) -> list[str]:
    """Lista os problemas do notebook; lista vazia = notebook seguro."""
    errors = []
    for number, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("attachments"):
            errors.append(f"Célula {number}: imagem anexada.")
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            for mime, value in data.items():
                text = _as_text(value)
                if mime.startswith("image/") and mime != "image/svg+xml":
                    errors.append(f"Célula {number}: imagem em bitmap ({mime}).")
                elif BITMAP_RE.search(text):
                    errors.append(f"Célula {number}: bitmap embutido ({mime}).")
                elif CLIENT_TEXT_RE.search(text):
                    errors.append(f"Célula {number}: possível nome de arquivo do cliente.")
            if CLIENT_TEXT_RE.search(_as_text(output.get("text", ""))):
                errors.append(f"Célula {number}: possível nome de arquivo do cliente.")
    return errors


def _cell(**output) -> dict:
    return {"cells": [{"cell_type": "code", "outputs": [output]}]}


SVG_CHART = "<svg><path d='M0 0L10 10'/></svg>"


def test_clean_notebook_passes():
    notebook = _cell(data={"image/svg+xml": SVG_CHART, "text/plain": "F-79b380974881"})
    assert check_notebook(notebook) == []


@pytest.mark.parametrize("notebook", [
    _cell(data={"image/png": "iVBORw0KGgo="}),                            # bitmap
    _cell(data={"image/svg+xml": "<svg><image href='data:image/jpeg'/></svg>"}),
    _cell(data={"text/html": "<img src='foto.png'>"}),                    # HTML
    _cell(name="stdout", text="lendo foto_001.JPG"),                     # nome real
    _cell(data={"text/plain": "12345678901234567890"}),                   # 20 dígitos
    {"cells": [{"cell_type": "markdown", "attachments": {"a.png": {}}}]},  # anexo
])
def test_unsafe_notebook_fails(notebook):
    assert check_notebook(notebook)


def test_versioned_notebooks():
    listed = subprocess.run(
        ["git", "ls-files", "*.ipynb"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.split()
    for name in listed:
        notebook = json.loads((REPO_ROOT / name).read_text(encoding="utf-8"))
        assert check_notebook(notebook) == [], name  # o nome do notebook é nosso

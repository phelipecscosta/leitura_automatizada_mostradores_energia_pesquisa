# Gera os diagramas das arquiteturas "B" e "A"
# Versão 3.2 (17/11/2026), decisões da ET4f, da ET5a e da ET5b:
# - triagem de cinco classes (digital, ciclométrico, indeterminado, múltiplos, outros) treinada com
#   rótulos finos; modelo do produto: C4, MobileNetV3-Large destilada (ET4f);
# - legibilidade retirada da triagem; Roteador de triagem com três saídas de rejeição;
# - detector SSDlite + MobileNetV3 na resolução nativa, com quatro classes de caixa (ET5a e ET5b),
#   rodando em CPU ou GPU com o mesmo modelo;
# - segunda porta do indeterminado: nenhum campo (display ou número de série) acima do limiar;
# - kvarh descartado por posição (o display mais próximo do número de série);
# - Validador 2: tipo da triagem contra a família do display mais confiante;
# - leitor com três faixas de confiança (muito baixa, intermediária e alta);
# - nova pasta "Rejeitadas / Múltiplos";
# - arquitetura A: detector único e leitores especializados por família (ablação da ET4c).
# A numeração dos módulos (B1 a B15 e A1 a A18) foi mantida; as novas decisões pertencem a módulos existentes.
# Uso: python gerar_arquiteturas.py  (salva PNG e JPG na pasta atual)
# Dependências: matplotlib e pillow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, PathPatch
from matplotlib.path import Path
from matplotlib import font_manager as fm

# Fonte: Carlito (Linux) ou Calibri (Windows), métricas iguais; DejaVu Sans como último recurso
for f in ["Regular", "Bold", "Italic", "BoldItalic"]:
    try:
        fm.fontManager.addfont(f"/usr/share/fonts/truetype/crosextra/Carlito-{f}.ttf")
    except Exception:
        pass


def escolher_fonte():
    disponiveis = {f.name for f in fm.fontManager.ttflist}
    for nome in ("Carlito", "Calibri", "DejaVu Sans"):
        if nome in disponiveis:
            return nome
    return "sans-serif"


FONTE = escolher_fonte()
plt.rcParams["font.family"] = FONTE
print(f"Fonte usada: {FONTE}")

AZUL = "#29A3E0"; SETA = "#4472C4"; VERDE = "#00A443"; TITULO = "#FFC000"
CINZA = "#595959"; ROTULO = "#7F7F7F"; VERMELHO = "#D74343"; FUNDO_APR = "#E3F2FC"; TEXTO = "#1E2822"

X_REJ = 17        # centro das caixas de rejeição (lado esquerdo)
W_REJ = 22        # largura das caixas de rejeição
X_BUS_REJ = 3     # barramento vertical das rejeições


class Canvas:
    """Área de desenho com y crescendo para cima; a altura final é ajustada ao conteúdo."""

    def __init__(self, largura):
        self.largura = largura
        self.fig = plt.figure(figsize=(largura / 10, 40))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.axis("off")

    def caixa(self, x, y, w, h, txt, borda=AZUL, fundo="white", lw=1.6, fs=10.5, peso="normal"):
        self.ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="square,pad=0",
                                         ec=borda, fc=fundo, lw=lw, zorder=3))
        self.ax.text(x, y, txt, ha="center", va="center", fontsize=fs, weight=peso, zorder=4)

    def aprendido(self, x, y, w, h, txt, fs=10.5):
        self.caixa(x, y, w, h, txt, fundo=FUNDO_APR, lw=2.6, fs=fs)

    def losango(self, x, y, w, h, txt, fs=10):
        pts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
        self.ax.add_patch(Polygon(pts, closed=True, ec=AZUL, fc="white", lw=1.8, zorder=3))
        if txt:
            self.ax.text(x, y, txt, ha="center", va="center", fontsize=fs, zorder=4)

    def losango_limiar(self, x, y, w, h, linha1, antes, depois="?"):
        # Losango com o limiar 0.XX destacado em vermelho na segunda linha
        self.losango(x, y, w, h, "")
        self.ax.text(x, y + 1.2, linha1, ha="center", va="center", fontsize=10, zorder=5)
        self.ax.text(x + 3.2, y - 1.3, antes, ha="right", va="center", fontsize=10, zorder=5)
        self.ax.text(x + 3.2, y - 1.3, "0.XX", ha="left", va="center", fontsize=10, color=VERMELHO, zorder=5)
        self.ax.text(x + 6.2, y - 1.3, depois, ha="left", va="center", fontsize=10, zorder=5)

    def modulo(self, x, y, txt, ha="left"):
        self.ax.text(x, y, txt, ha=ha, va="center", fontsize=10, style="italic", color=VERDE, zorder=4)

    def rotulo(self, x, y, txt, ha="center", rot=0):
        self.ax.text(x, y, txt, ha=ha, va="center", fontsize=9.5, style="italic", color=ROTULO,
                     rotation=rot, zorder=4)

    def seta(self, pontos):
        xs = [p[0] for p in pontos]; ys = [p[1] for p in pontos]
        if len(pontos) > 2:
            self.ax.plot(xs[:-1], ys[:-1], color=SETA, lw=1.5, zorder=2)
        self.ax.annotate("", xy=pontos[-1], xytext=pontos[-2],
                         arrowprops=dict(arrowstyle="-|>", color=SETA, lw=1.5, mutation_scale=13), zorder=2)

    def linha(self, pontos):
        self.ax.plot([p[0] for p in pontos], [p[1] for p in pontos], color=SETA, lw=1.5, zorder=2)

    def chave(self, x, y, h):
        v = [(x + 1.2, y + h / 2), (x, y + h / 2), (x, y + h / 2 - 1.2), (x, y + 1.0), (x, y), (x - 1.2, y),
             (x, y), (x, y - 1.0), (x, y - h / 2 + 1.2), (x, y - h / 2), (x + 1.2, y - h / 2)]
        c = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3]
        self.ax.add_patch(PathPatch(Path(v, c), fill=False, ec=SETA, lw=1.5, zorder=2))

    def lista(self, x_chave, y, itens, w=27, h=3.4, gap=0.9, fs=9.5, abertos=()):
        """Lista de saídas à direita de um módulo; os índices em 'abertos' recebem borda de decisão em aberto."""
        total = len(itens) * h + (len(itens) - 1) * gap
        self.chave(x_chave, y, total + 1.5)
        topo = y + total / 2 - h / 2
        for i, t in enumerate(itens):
            kw = dict(borda=VERMELHO, lw=2) if i in abertos else dict(borda=CINZA)
            self.caixa(x_chave + 2.2 + w / 2, topo - i * (h + gap), w, h, t, fs=fs, **kw)

    def titulo(self, x, y, txt):
        self.ax.text(x, y, txt, ha="center", va="center", fontsize=17, weight="bold", color=TITULO)

    def legenda(self, lx, ly):
        self.ax.text(lx, ly, "Legenda", fontsize=10, weight="bold", color=TEXTO, va="center")
        itens = [("Modelo aprendido", dict(borda=AZUL, fundo=FUNDO_APR, lw=2.6)),
                 ("Regra ou processamento determinístico", dict(borda=AZUL, fundo="white", lw=1.6)),
                 ("Arquivos de entrada e saída", dict(borda=CINZA, fundo="white", lw=1.6)),
                 ("Decisão em aberto", dict(borda=VERMELHO, fundo="white", lw=2))]
        for i, (t, kw) in enumerate(itens):
            yy = ly - 3.2 * (i + 1)
            self.caixa(lx + 2.5, yy, 5, 2.2, "", **kw)
            self.ax.text(lx + 6.5, yy, t, fontsize=9.5, va="center", color=TEXTO)
        return ly - 3.2 * len(itens) - 1.5

    def salvar(self, nome, y0, y1):
        # Ajusta a figura à altura usada, mantendo 1 unidade = 1/10 de polegada nos dois eixos
        self.fig.set_size_inches(self.largura / 10, (y1 - y0) / 10)
        self.ax.set_xlim(0, self.largura); self.ax.set_ylim(y0, y1)
        self.ax.set_aspect("equal")
        self.fig.savefig(nome + ".png", dpi=150, facecolor="white")
        from PIL import Image
        Image.open(nome + ".png").convert("RGB").save(nome + ".jpg", quality=95)
        plt.close(self.fig)


class Coluna:
    """Fluxo principal de cima para baixo; guarda as saídas para os barramentos."""

    def __init__(self, cv, X, y_topo):
        self.cv, self.X, self.y = cv, X, y_topo   # self.y = base do último elemento desenhado
        self.ys_rej = []                           # saídas para o barramento das rejeições
        self.verificar = []                        # saídas para o barramento de "Verificar"

    def proximo(self, h, gap=4.0, rotulo=None):
        """Desenha a seta até o próximo elemento de altura h e devolve o centro dele."""
        topo = self.y - gap
        self.cv.seta([(self.X, self.y), (self.X, topo)])
        if rotulo:
            self.cv.rotulo(self.X + 1.5, self.y - gap / 2, rotulo, ha="left")
        centro = topo - h / 2
        self.y = centro - h / 2
        return centro

    def rejeitar(self, y, x_saida, txt, rotulo, h=5.5, fs=10):
        self.cv.caixa(X_REJ, y, W_REJ, h, txt, fs=fs)
        self.cv.seta([(x_saida, y), (X_REJ + W_REJ / 2, y)])
        self.cv.rotulo((x_saida + X_REJ + W_REJ / 2) / 2, y + 1.5, rotulo)
        self.ys_rej.append(y)

    def para_verificar(self, x_saida, y, rotulo):
        self.verificar.append((x_saida, y, rotulo))


def topo(cv, col, P):
    """Trecho comum: ingestão, integridade, pareamento, listagem na BaseExtracao, pré-processamento I,
    triagem, roteador de triagem e filtro de Parâmetros. P é o prefixo do módulo (A ou B)."""
    ax = cv.ax; X = col.X

    # Ingestão
    y = col.y - 3.5
    ax.add_patch(FancyBboxPatch((X - 13, y - 3.5), 26, 7, boxstyle="round,pad=0,rounding_size=1.8",
                                ec=VERDE, fc=VERDE, lw=1, zorder=3))
    ax.text(X, y, "Ingestão", ha="center", va="center", fontsize=16, weight="bold", color="white", zorder=4)
    cv.modulo(X + 15, y + 1.5, "Imagens do lote")
    col.y = y - 3.5

    # Integridade (Módulo 1)
    y = col.proximo(9)
    cv.losango(X, y, 22, 9, "Arquivo íntegro?")
    cv.modulo(X + 13, y + 3, f"Módulo {P}1 - Validador 1 (regra)")
    col.rejeitar(y, X - 11, "Rejeitar -\nCorrompidas", "Não")

    # Pareamento (Módulo 2)
    y = col.proximo(6, rotulo="Sim")
    cv.caixa(X, y, 26, 6, "Pareamento foto ↔ CSV\n(número esperado, leitura digitada)", fs=10)
    cv.modulo(X - 14, y + 4.5, f"Módulo {P}2 - Pareador (regra)", ha="right")
    cv.caixa(X + 34, y, 22, 6, "BaseExtracao_\nDDMMAAAA_Dia.csv", borda=CINZA, fs=10)
    cv.seta([(X + 23, y), (X + 13, y)])
    cv.modulo(X + 34, y + 4.5, "Tabela de controle", ha="center")

    # Foto listada na BaseExtracao? (decisão do Módulo 2)
    y = col.proximo(10)
    cv.losango(X, y, 26, 10, "Foto listada na\nBaseExtracao?")
    cv.modulo(X + 14, y, f"(decisão do Módulo {P}2)")
    col.rejeitar(y, X - 13, "Rejeitar -\nNão listada", "Não")

    # Pré-processamento I (Módulo 3)
    y = col.proximo(5, rotulo="Sim")
    cv.caixa(X, y, 26, 5, "Pré-processamento I")
    cv.modulo(X - 14, y, f"Módulo {P}3 - Pré-processador 1", ha="right")

    # Triagem (Módulo 4): cinco classes do produto, treino com rótulos finos
    y = col.proximo(7, gap=6)
    cv.aprendido(X, y, 26, 7, "Triagem\n(5 classes do produto)")
    cv.modulo(X - 14, y + 1.4, f"Módulo {P}4 - Classificador 1", ha="right")
    cv.modulo(X - 14, y - 1.4, "(soma das probabilidades dos grupos)", ha="right")
    cv.linha([(X + 13, y), (X + 14.3, y)])
    cv.lista(X + 15.5, y, ["digital | ciclométrico", "indeterminado | múltiplos | outros",
                           "treino: rótulos finos (grupos)", "modelo: C4 (MobileNetV3-Large)"],
             w=31, h=3.0, gap=0.7)

    # Roteador de triagem (Módulo 5): três saídas de rejeição, com limiares calibrados
    y = col.proximo(9, gap=6)
    cv.losango(X, y, 22, 9, "É medidor?")
    cv.modulo(X + 12, y + 1.5, f"Módulo {P}5 - Roteador de triagem")
    cv.modulo(X + 12, y - 1.5, "(regras sobre a triagem calibrada)")
    col.rejeitar(y, X - 11, "Rejeitar - Outros", "Não", h=5)

    y = col.proximo(10, rotulo="Sim")
    cv.losango_limiar(X, y, 28, 10, "Indeterminado", "com P ≥ ")
    cv.modulo(X + 15, y, f"(decisão do Módulo {P}5)")
    col.rejeitar(y, X - 14, "Rejeitar - Baixa qualidade\n(Indeterminação do\ntipo de medidor)", "Sim", h=8, fs=9.5)

    y = col.proximo(10, rotulo="Não")
    cv.losango_limiar(X, y, 28, 10, "Múltiplos", "com P ≥ ")
    cv.modulo(X + 15, y, f"(decisão do Módulo {P}5)")
    col.rejeitar(y, X - 14, "Rejeitar -\nMúltiplos medidores", "Sim", h=6, fs=9.5)

    # Medidor listado nos Parâmetros? (Módulo 6, só atua quando a planilha é fornecida)
    y = col.proximo(10, rotulo="Não")
    cv.losango(X, y, 26, 10, "Medidor listado\nnos Parâmetros?")
    cv.modulo(X - 14, y + 6.5, f"Módulo {P}6 - Filtro de Parâmetros (regra)", ha="right")
    col.rejeitar(y, X - 13, "Não processada\n(permanece na origem)", "Não", h=6, fs=9.5)
    cv.caixa(X + 34, y, 22, 6, "Planilha de\nParâmetros (.xlsx)", borda=CINZA, fs=10)
    cv.seta([(X + 23, y), (X + 13, y)])
    cv.modulo(X + 34, y + 4.5, "Opcional", ha="center")


DETECTOR_CLASSES = ["display digital (consumo + função)", "display ciclométrico (consumo)",
                    "display kvarh (descartado)", "número de série"]


def detector_e_tipo(cv, col, P):
    """Detector único, segunda porta do indeterminado, descarte do kvarh e concordância de tipo
    (comum a A e B)."""
    X = col.X
    y = col.proximo(6.5, rotulo="Sim (ou sem planilha)")
    cv.aprendido(X, y, 26, 6.5, "Detecção de regiões")
    cv.modulo(X - 14, y + 2.8, f"Módulo {P}7 - Detector 1", ha="right")
    cv.modulo(X - 14, y, "SSDlite + MobileNetV3", ha="right")
    cv.modulo(X - 14, y - 2.8, "(360 × 480; CPU ou GPU)", ha="right")
    cv.linha([(X + 13, y), (X + 14.3, y)])
    cv.lista(X + 15.5, y, DETECTOR_CLASSES, w=29, h=3.0, gap=0.7)

    # Segunda porta do indeterminado: nenhum campo (display ou número de série) acima do limiar
    y = col.proximo(10, gap=6)
    cv.losango_limiar(X, y, 28, 10, "Campo localizado", "com conf. ≥ ")
    cv.modulo(X + 15, y, f"(decisão do Módulo {P}7)")
    col.rejeitar(y, X - 14, "Rejeitar - Baixa qualidade\n(Indeterminação do\ntipo de medidor)", "Não", h=8, fs=9.5)

    # Descarte do kvarh por posição, e não pela classe prevista
    y = col.proximo(7, rotulo="Sim")
    cv.caixa(X, y, 30, 7, "Descarte do kvarh\n(display mais próximo\ndo número de série)", fs=10)
    cv.modulo(X - 16, y, f"(regra do Módulo {P}7)", ha="right")

    y = col.proximo(11)
    cv.losango(X, y, 30, 11, "Tipo da triagem =\ndisplay mais confiante?")
    cv.modulo(X - 16, y + 4.5, f"Módulo {P}8 - Validador 2 (regra)", ha="right")
    col.para_verificar(X + 15, y, "Não")


def faixas_e_validadores(cv, col, P, n, origem, rotulo_entrada=None):
    """Faixas de confiança do leitor e validadores 3 e 4; n é o número do Validador 3."""
    X = col.X
    y = col.proximo(12, rotulo=rotulo_entrada)
    cv.losango(X, y, 30, 12, "Faixa de confiança\nda leitura?")
    cv.modulo(X - 16, y + 6, f"({origem}; limiares 0.XX)", ha="right")
    col.rejeitar(y, X - 15, "Rejeitar - Baixa qualidade\n(Falha na extração)", "Muito baixa", h=6, fs=9.5)
    col.para_verificar(X + 15, y, "Intermediária")

    y = col.proximo(13, rotulo="Alta")
    cv.losango_limiar(X, y, 36, 13, "Esperado = leitura livre", "e confiança ≥ ")
    cv.modulo(X - 19, y + 5, f"Módulo {P}{n} - Validador 3 (regra)", ha="right")
    cv.modulo(X - 19, y + 2, "número e leitura vs. CSV", ha="right")
    col.para_verificar(X + 18, y, "Não")

    y = col.proximo(14, rotulo="Sim")
    cv.losango(X, y, 40, 14, "Função = 03 (digital) e\nleitura nos Parâmetros\ne criticidade B?", fs=9.5)
    cv.modulo(X - 21, y + 4, f"Módulo {P}{n + 1} - Validador 4 (regra)", ha="right")
    cv.modulo(X - 21, y + 1, "(Parâmetros: só com planilha)", ha="right")
    col.para_verificar(X + 20, y, "Não")


def base(cv, col, P, n, xb):
    """Trecho comum final: planilha, barramentos, courier, pastas, feedback e legenda.
    n é o número do módulo da planilha de controle. Devolve o y mais baixo desenhado."""
    X = col.X; largura = cv.largura
    y = col.proximo(6, rotulo="Sim")
    cv.caixa(X, y, 40, 6, "Preenchimento automático da planilha de controle\n(status + motivo, todas as imagens)", fs=10)
    cv.modulo(X - 21, y + 4.2, f"Módulo {P}{n} - Planilha de controle", ha="right")

    # Barramento de "Verificar" (direita)
    for (x0, yy, rot) in col.verificar:
        cv.linha([(x0, yy), (xb, yy)]); cv.rotulo(x0 + 3, yy + 1.5, rot, ha="left")
    y_top = max(yy for _, yy, _ in col.verificar)
    cv.linha([(xb, y_top), (xb, y)])
    cv.seta([(xb, y), (X + 20, y)])
    cv.rotulo(xb + 1.6, (y_top + y) / 2, "Necessita de verificação", rot=90)

    # Barramento das rejeições (esquerda)
    for yy in col.ys_rej:
        cv.linha([(X_REJ - W_REJ / 2, yy), (X_BUS_REJ, yy)])
    y_rej = max(col.ys_rej)
    cv.linha([(X_BUS_REJ, y_rej), (X_BUS_REJ, y)])
    cv.seta([(X_BUS_REJ, y), (X - 20, y)])
    cv.rotulo(X_BUS_REJ - 1.9, (y_rej + y) / 2, "Rejeições e não processadas (motivo registrado)", rot=90)

    # Courier e pastas (espaçamento e largura calculados pelo número de pastas)
    yc = y - 5
    cv.seta([(X, y - 3), (X, yc)])
    nomes = ["Aprovadas", "Verificar", "Rejeitadas /\nOutros", "Rejeitadas /\nBaixa qualidade",
             "Rejeitadas /\nMúltiplos", "Rejeitadas /\nCorrompidas", "Rejeitadas /\nNão listada"]
    passo = (largura - 20) / (len(nomes) - 1)       # distância entre os centros das pastas
    xs = [10 + i * passo for i in range(len(nomes))]
    w_pasta = min(21, passo - 1.6)                   # largura da caixa, com folga entre elas
    cv.linha([(xs[0], yc), (xs[-1], yc)])
    for xp, t in zip(xs, nomes):
        cv.seta([(xp, yc), (xp, yc - 1.5)])
        cv.caixa(xp, yc - 4, w_pasta, 5, t, borda=CINZA, fs=9)
    cv.modulo(X + 2, yc + 1.1, f"Módulo {P}{n + 1} - Courier")

    # Feedback: sai do vão entre a 4ª e a 5ª pasta, para não cruzar nenhuma caixa nem a legenda
    xf = (xs[3] + xs[4]) / 2
    cv.seta([(xf, yc), (xf, yc - 11)])
    cv.caixa(xf, yc - 14.5, 30, 6, "Feedback com estatísticas\ndo processamento", borda=CINZA, fs=10)
    cv.modulo(xf + 16, yc - 14.5, f"Módulo {P}{n + 2} - Feedback")
    return min(cv.legenda(3, yc - 10), yc - 19)


# ---------------- Arquitetura B ----------------
cv = Canvas(104); col = Coluna(cv, 50, -6)
cv.titulo(47, -2.5, "Arquitetura B (Multiclasse)")
topo(cv, col, "B")
detector_e_tipo(cv, col, "B")

X = col.X
y = col.proximo(6, rotulo="Sim")
cv.caixa(X, y, 30, 6, "Pré-processamento II\n(recorte, cinza, retificação)", fs=10)
cv.modulo(X - 16, y, "Módulo B9 - Pré-processador 2", ha="right")

y = col.proximo(7)
cv.aprendido(X, y, 30, 7, "Leitura com verificação\n(CRNN + CTC)")
cv.modulo(X - 16, y, "Módulo B10 - Leitor", ha="right")
cv.linha([(X + 15, y), (X + 16.3, y)])
cv.lista(X + 17.5, y, ["leitura livre", "prob. da sequência esperada", "regra do dígito em transição",
                       "confiança por campo"], w=24, h=3.0, gap=0.7, fs=9)

faixas_e_validadores(cv, col, "B", 11, "decisão do Módulo B10", rotulo_entrada=None)
y_min = base(cv, col, "B", 13, xb=101)
cv.salvar("13_arquitetura_B_(Multiclasse)", y_min - 2, 0)

# ---------------- Arquitetura A ----------------
# Detector único (as mesmas caixas e recortes da B) e leitores especializados por família:
# é a ablação do leitor por família decidida na ET4c.
cv = Canvas(136); col = Coluna(cv, 66, -6)
PI, PII = 38, 94
cv.titulo(60, -2.5, "Arquitetura A (Divisão de classes)")
topo(cv, col, "A")
detector_e_tipo(cv, col, "A")

X = col.X
y = col.proximo(10, rotulo="Sim")
cv.losango(X, y, 24, 10, "Qual o tipo?")
cv.modulo(X - 13, y + 4.5, "Módulo A9 - Roteador de tipo (regra)", ha="right")
cv.linha([(X - 12, y), (PI, y)])
cv.linha([(X + 12, y), (PII, y)])
cv.rotulo(X - 18, y + 1.5, "Digital"); cv.rotulo(X + 20, y + 1.5, "Ciclométrico")
cv.modulo(PI - 2, y - 2.5, "Pipeline I", ha="right"); cv.modulo(PII + 2, y - 2.5, "Pipeline II")

# Pré-processamento por família
y_pp = y - 10
cv.seta([(PI, y), (PI, y_pp + 3.5)]); cv.seta([(PII, y), (PII, y_pp + 3.5)])
cv.caixa(PI, y_pp, 24, 7, "Pré-processamento II\n(recorte, cinza,\nretificação)", fs=9.5)
cv.caixa(PII, y_pp, 24, 7, "Pré-processamento III\n(recorte, cinza,\nretificação)", fs=9.5)
cv.modulo(PI - 13.5, y_pp, "Módulo A10 - Pré-processador 2", ha="right")
cv.modulo(PII + 13.5, y_pp, "Módulo A12 - Pré-processador 3")

# Leitores por família
y_l = y_pp - 12.5
cv.seta([(PI, y_pp - 3.5), (PI, y_l + 4)]); cv.seta([(PII, y_pp - 3.5), (PII, y_l + 4)])
cv.aprendido(PI, y_l, 24, 8, "Leitor 1 (7 segmentos)\ncom verificação", fs=10)
cv.aprendido(PII, y_l, 24, 8, "Leitor 2 (tambores)\ncom verificação", fs=10)
cv.modulo(PI - 13.5, y_l + 1.4, "Módulo A11 - Leitor 1", ha="right")
cv.modulo(PI - 13.5, y_l - 1.4, "(CRNN + CTC)", ha="right")
cv.modulo(PII + 13.5, y_l + 9.6, "Módulo A13 - Leitor 2 (CRNN + CTC)")
cv.linha([(PI + 12, y_l), (PI + 13.3, y_l)])
cv.lista(PI + 14.5, y_l, ["leitura livre", "prob. da sequência esperada", "confiança por campo"],
         w=19, h=3.0, gap=0.7, fs=9)
cv.linha([(PII + 12, y_l), (PII + 13.3, y_l)])
cv.lista(PII + 14.5, y_l, ["leitura livre", "prob. da sequência esperada", "regra do dígito em transição",
                           "confiança por campo"], w=19, h=3.0, gap=0.7, fs=9)

# Junção dos pipelines
y_j = y_l - 4 - 3.5
cv.linha([(PI, y_l - 4), (PI, y_j), (X, y_j)])
cv.linha([(PII, y_l - 4), (PII, y_j), (X, y_j)])
col.y = y_j

faixas_e_validadores(cv, col, "A", 14, "decisão dos Módulos A11 e A13")
y_min = base(cv, col, "A", 16, xb=133)
cv.salvar("12_arquitetura_A_(Divisao_de_classes)", y_min - 2, 0)
print("Diagramas gerados na pasta atual.")

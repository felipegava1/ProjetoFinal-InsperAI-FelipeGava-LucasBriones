"""Caminhos e constantes comuns a todas as etapas (E0 a E5) do classificador de dermatoscopia."""
import os
from pathlib import Path

# Raiz dos dados (fora do repositório). Mude com a variável de ambiente DERMATO_DATA.
DATA = Path(os.environ.get("DERMATO_DATA", r"F:\dermato\data"))
RAW = DATA / "raw"
EXTRAIDO = DATA / "extracted"
CACHE = DATA / "cache_448"          # menor lado 448 px, sem Shades of Gray (aplicado na hora)
CACHE_LADO = 448

REPO = Path(__file__).resolve().parent.parent
SPLITS = REPO / "splits" / "dermato.csv"                       # divisão fixa (versionada)
RESULTADOS = Path(os.environ.get("DERMATO_RESULTADOS", DATA.parent / "resultados"))
EXPERIMENTOS = REPO / "resultados" / "experimentos.csv"        # uma linha por rodada (versionado)


def achar(nome: str, sub: str) -> Path:
    """Acha um arquivo de resultado (ex.: achar("E1_final_s42.pt", "modelos")) procurando na pasta de
    resultados desta máquina (DERMATO_RESULTADOS) e na pasta resultados/ do repositório. Assim quem
    clona o repositório só precisa colocar os pesos em resultados/modelos/."""
    for base in (RESULTADOS, REPO / "resultados"):
        if (base / sub / nome).exists():
            return base / sub / nome
    raise FileNotFoundError(f"{nome} não encontrado em {RESULTADOS / sub} nem em {REPO / 'resultados' / sub}")

SEED = 42
CLASSES = ["ak", "bcc", "bkl", "df", "mel", "nv", "scc", "vasc"]
IDX = {c: i for i, c in enumerate(CLASSES)}
MALIGNAS = ["ak", "bcc", "mel", "scc"]   # ak conta como maligna, como o akiec do HAM10000
MAL_IDX = [IDX[c] for c in MALIGNAS]

# Teste T1 (ISIC 2018) só tem as 7 classes do HAM10000: akiec = ak + scc
HAM_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
PARA_HAM = {"ak": "akiec", "scc": "akiec", "bcc": "bcc", "bkl": "bkl", "df": "df",
            "mel": "mel", "nv": "nv", "vasc": "vasc"}

SITES = ["anterior torso", "head/neck", "lateral torso", "lower extremity",
         "oral/genital", "palms/soles", "posterior torso", "upper extremity"]
N_META = 2 + 3 + len(SITES) + 1   # idade + faltante, sexo (f, m, faltante), região (8 + faltante)

"""Baixa as fotos clínicas listadas por baixar_metadados_clinicas.py, exceto as que já
vieram no zip da MILK10k.

Uso:  py -3.12 scripts/baixar_imagens_clinicas.py
Saída: data/raw/isic_clinicas/imagens/<isic_id>.jpg
"""
import csv, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PASTA = RAIZ / "data" / "raw" / "isic_clinicas"
IMAGENS = PASTA / "imagens"

def baixar(linha):
    destino = IMAGENS / f"{linha['isic_id']}.jpg"
    # A API informa um tamanho ~3 KB menor que o arquivo servido, então conferimos só
    # a assinatura (JPEG ou PNG); a decodificação completa é feita em preparar_dados.py.
    if destino.exists() and destino.stat().st_size > 0:
        return True
    for tentativa in range(1, 6):
        try:
            with urllib.request.urlopen(linha["url_full"], timeout=120) as r:
                dados = r.read()
            if not (dados.startswith(b"\xff\xd8") or dados.startswith(b"\x89PNG")):
                raise ValueError(f"resposta não é imagem ({len(dados)} bytes)")
            destino.write_bytes(dados)
            return True
        except Exception as erro:
            print(f"falha {linha['isic_id']} ({tentativa}): {erro}", flush=True); time.sleep(3 * tentativa)
    return False

def main():
    IMAGENS.mkdir(parents=True, exist_ok=True)
    with open(PASTA / "metadados.csv", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    with open(RAIZ / "data/raw/milk10k/MILK10k_Training_Metadata.csv") as f:
        ja_na_milk = {l["isic_id"] for l in csv.DictReader(f)}
    pendentes = [l for l in linhas if l["isic_id"] not in ja_na_milk]
    print(f"{len(pendentes)} imagens a baixar ({sum(int(l['bytes_full']) for l in pendentes)/1e9:.1f} GB)", flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        ok = list(ex.map(baixar, pendentes))
    print(f"baixadas {sum(ok)} de {len(pendentes)}")

if __name__ == "__main__":
    main()

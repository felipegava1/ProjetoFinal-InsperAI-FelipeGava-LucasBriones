"""Baixa do ISIC Archive só as imagens do ISIC 2020 que entram no projeto (diagnóstico específico).

São ~6 mil das 33 mil imagens do zip oficial de 24,7 GB. Originais em DATA/extracted/isic2020_api/.
Uso: python scripts_dermato/baixar_isic2020.py
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dermato.config import EXTRAIDO, SPLITS  # noqa: E402

DEST = EXTRAIDO / "isic2020_api"
URL = "https://isic-archive.s3.us-east-1.amazonaws.com/images/{}.jpg"


def baixar(isic_id, sessao=requests.Session()):
    alvo = DEST / f"{isic_id}.jpg"
    if alvo.exists() and alvo.stat().st_size > 0:
        return 0
    for _ in range(5):
        try:
            r = sessao.get(URL.format(isic_id), timeout=60)
            if r.status_code == 200 and r.content[:2] == b"\xff\xd8":
                tmp = alvo.with_suffix(".tmp")
                tmp.write_bytes(r.content)
                tmp.replace(alvo)
                return len(r.content)
        except requests.RequestException:
            pass
    print("FALHOU", isic_id, flush=True)
    return -1


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SPLITS)
    ids = df.loc[df["base"] == "isic2020", "isic_id"].tolist()
    print(f"{len(ids)} imagens do ISIC 2020", flush=True)
    total = falhas = 0
    with ThreadPoolExecutor(16) as ex:
        for k, n in enumerate(ex.map(baixar, ids)):
            falhas += n < 0
            total += max(n, 0)
            if k % 500 == 0:
                print(f"  {k}/{len(ids)}  {total / 1e9:.2f} GB", flush=True)
    print(f"Pronto: {total / 1e9:.2f} GB, falhas {falhas}")
    sys.exit(1 if falhas else 0)


if __name__ == "__main__":
    main()

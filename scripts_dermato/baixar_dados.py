"""Baixa e extrai as bases públicas usadas pelo classificador de dermatoscopia (~15 GB).

Uso:
    python scripts_dermato/baixar_dados.py            # tudo
    python scripts_dermato/baixar_dados.py --sem-treino   # só o necessário para avaliar (T1/T2)

Destino: DERMATO_DATA (padrão F:\\dermato\\data) em raw/ (arquivos baixados) e extracted/ (imagens).
Retomável: arquivos já completos são pulados. Depois rode:
    python scripts_dermato/baixar_isic2020.py     # só as 5.993 imagens do ISIC 2020 usadas
    python -m dermato.preparar cache              # redimensiona tudo para o cache de 448 px
Licenças: ISIC e MILK10k CC BY-NC 4.0 (uso não comercial).
"""
import argparse
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dermato.config import EXTRAIDO, RAW  # noqa: E402

C = "https://isic-archive.s3.amazonaws.com/challenges"
M = "https://isic-archive.s3.amazonaws.com/dois/10.34970-648456"
D = "https://dataverse.harvard.edu/api/access/datafile"
AVALIACAO = {   # rótulos e imagens do T1 (ISIC 2018) e do T2 (teste ISIC 2019) + dermatologistas
    "ISIC_2019_Test_GroundTruth.csv": f"{C}/2019/ISIC_2019_Test_GroundTruth.csv",
    "ISIC_2019_Test_Metadata.csv": f"{C}/2019/ISIC_2019_Test_Metadata.csv",
    "ISIC2018_Task3_Test_GroundTruth.zip": f"{C}/2018/ISIC2018_Task3_Test_GroundTruth.zip",
    "ISIC2018_Task3_Test_NatureMedicine_AI_Interaction_Benefit.csv": f"{D}/3864681?format=original",
    "ISIC2018_Task3_Test_Input.zip": f"{C}/2018/ISIC2018_Task3_Test_Input.zip",
    "ISIC_2019_Test_Input.zip": f"{C}/2019/ISIC_2019_Test_Input.zip",
}
TREINO = {
    "ISIC_2019_Training_GroundTruth.csv": f"{C}/2019/ISIC_2019_Training_GroundTruth.csv",
    "ISIC_2019_Training_Metadata.csv": f"{C}/2019/ISIC_2019_Training_Metadata.csv",
    "ISIC_2020_Training_GroundTruth_v2.csv": f"{C}/2020/ISIC_2020_Training_GroundTruth_v2.csv",
    "ISIC_2020_Training_Duplicates.csv": f"{C}/2020/ISIC_2020_Training_Duplicates.csv",
    "milk10k_training_gt.csv": f"{M}/supplements/training_gt.csv",
    "milk10k_training_input.csv": f"{M}/supplements/training_input.csv",
    "milk10k_training_supp.csv": f"{M}/supplements/training_supp.csv",
    "milk10k.zip": f"{M}/milk10k.zip",
    "ISIC_2019_Training_Input.zip": f"{C}/2019/ISIC_2019_Training_Input.zip",
}


def baixar(nome, url):
    alvo = RAW / nome
    with requests.get(url, stream=True, timeout=60, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
    if alvo.exists() and total and alvo.stat().st_size == total:
        print(f"  ok (já baixado) {nome}")
        return alvo
    feito = alvo.stat().st_size if alvo.exists() else 0
    cab = {"Range": f"bytes={feito}-"} if feito else {}
    with requests.get(url, stream=True, timeout=60, headers=cab) as r:
        r.raise_for_status()
        modo = "ab" if feito and r.status_code == 206 else "wb"
        with open(alvo, modo) as f:
            for k, bloco in enumerate(r.iter_content(1 << 20)):
                f.write(bloco)
                if k % 500 == 0 and total:
                    print(f"  {nome}: {alvo.stat().st_size / 1e9:.2f}/{total / 1e9:.2f} GB", flush=True)
    print(f"  baixado {nome} ({alvo.stat().st_size / 1e6:.0f} MB)")
    return alvo


def extrair(zip_):
    destino = EXTRAIDO / zip_.stem
    if (EXTRAIDO / f"{zip_.stem}.ok").exists():
        return
    print(f"  extraindo {zip_.name} ...", flush=True)
    with zipfile.ZipFile(zip_) as z:
        z.extractall(destino)
    (EXTRAIDO / f"{zip_.stem}.ok").touch()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem-treino", action="store_true", help="só as bases de avaliação (T1, T2)")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    EXTRAIDO.mkdir(parents=True, exist_ok=True)
    lista = AVALIACAO | ({} if a.sem_treino else TREINO)
    for nome, url in lista.items():
        arq = baixar(nome, url)
        if nome.endswith(".zip"):
            extrair(arq)
    print("Pronto. Próximos passos: baixar_isic2020.py (se for treinar) e python -m dermato.preparar cache")


if __name__ == "__main__":
    main()

"""Baixa a base SCIN (Google, fotos de celular de condições de pele, licença SCIN Data Use License).

Uso:  py -3.12 scripts/baixar_scin.py
Saída: data/raw/scin/{scin_cases.csv, scin_labels.csv, scin_*_questions.csv, LICENSE, imagens/<id>.png}

Fonte: bucket público dx-scin-public-data (https://github.com/google-research-datasets/scin).
Retomável: imagens já baixadas e com assinatura PNG válida são puladas.
"""
import csv, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PASTA = RAIZ / "data" / "raw" / "scin"
IMAGENS = PASTA / "imagens"
BUCKET = "https://storage.googleapis.com/dx-scin-public-data/"
CSVS = ["scin_cases.csv", "scin_labels.csv", "scin_app_questions.csv", "scin_label_questions.csv"]


def baixar(caminho_no_bucket):
    destino = IMAGENS / Path(caminho_no_bucket).name
    if destino.exists() and destino.stat().st_size > 0:
        return True
    for tentativa in range(1, 6):
        try:
            with urllib.request.urlopen(BUCKET + caminho_no_bucket, timeout=120) as r:
                dados = r.read()
            if not (dados.startswith(b"\x89PNG") or dados.startswith(b"\xff\xd8")):
                raise ValueError(f"resposta não é imagem ({len(dados)} bytes)")
            destino.write_bytes(dados)
            return True
        except Exception as erro:
            print(f"falha {caminho_no_bucket} ({tentativa}): {erro}", flush=True)
            time.sleep(3 * tentativa)
    return False


def main():
    IMAGENS.mkdir(parents=True, exist_ok=True)
    for nome in CSVS:
        if not (PASTA / nome).exists():
            urllib.request.urlretrieve(BUCKET + "dataset/" + nome, PASTA / nome)
    if not (PASTA / "LICENSE").exists():
        urllib.request.urlretrieve("https://raw.githubusercontent.com/google-research-datasets/scin/main/LICENSE", PASTA / "LICENSE")
    with open(PASTA / "scin_cases.csv", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    caminhos = sorted({l[c] for l in linhas for c in ("image_1_path", "image_2_path", "image_3_path") if l.get(c)})
    print(f"{len(linhas)} casos, {len(caminhos)} imagens", flush=True)
    inicio = time.time()
    feitas = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        ok = []
        for r in ex.map(baixar, caminhos):
            ok.append(r)
            feitas += 1
            if feitas % 500 == 0:
                print(f"  {feitas}/{len(caminhos)}  {feitas / (time.time() - inicio):.1f} img/s", flush=True)
    print(f"baixadas {sum(ok)} de {len(caminhos)}; falhas: {len(ok) - sum(ok)}")


if __name__ == "__main__":
    main()

"""Baixa as bases rotuladas da ISIC (2018, 2019, 2020, 2024) e a MILK10k.

Uso:  py -3.12 scripts/baixar_dados.py [base ...]
      (sem argumentos baixa todas; bases: isic2018 isic2019 isic2020 isic2024 milk10k)

Os arquivos vão para data/raw/<base>/. O download retoma de onde parou se for
interrompido, e um arquivo só é considerado pronto quando o tamanho bate com o
do servidor.
"""

import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PASTA_RAW = RAIZ / "data" / "raw"
URL_BASE = "https://isic-archive.s3.amazonaws.com/challenges"

# Só arquivos com rótulo: os testes de 2020 e da MILK10k não têm gabarito público.
ARQUIVOS = {
    "isic2018": [  # Task 3 = HAM10000
        "2018/ISIC2018_Task3_Training_GroundTruth.zip",
        "2018/ISIC2018_Task3_Training_LesionGroupings.csv",
        "2018/ISIC2018_Task3_Validation_GroundTruth.zip",
        "2018/ISIC2018_Task3_Test_GroundTruth.zip",
        "2018/ISIC2018_Task3_Validation_Input.zip",
        "2018/ISIC2018_Task3_Test_Input.zip",
        "2018/ISIC2018_Task3_Training_Input.zip",
    ],
    "isic2019": [
        "2019/ISIC_2019_Training_GroundTruth.csv",
        "2019/ISIC_2019_Training_Metadata.csv",
        "2019/ISIC_2019_Test_GroundTruth.csv",
        "2019/ISIC_2019_Test_Metadata.csv",
        "2019/ISIC_2019_Test_Input.zip",
        "2019/ISIC_2019_Training_Input.zip",
    ],
    "isic2020": [
        "2020/ISIC_2020_Training_GroundTruth.csv",
        "2020/ISIC_2020_Training_GroundTruth_v2.csv",
        "2020/ISIC_2020_Training_Duplicates.csv",
        "2020/ISIC_2020_Training_JPEG.zip",
    ],
    "isic2024": [
        "2024/ISIC_2024_Training_GroundTruth.csv",
        "2024/ISIC_2024_Training_Supplement.csv",
        "2024/ISIC_2024_Training_Input.zip",
    ],
    "milk10k": [
        "milk10k/MILK10k_Training_GroundTruth.csv",
        "milk10k/MILK10k_Training_Metadata.csv",
        "milk10k/MILK10k_Training_Supplement.csv",
        "milk10k/MILK10k_Training_Input.zip",
    ],
}

BLOCO = 1 << 20  # 1 MB


def tamanho_remoto(url):
    pedido = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(pedido, timeout=60) as resposta:
        return int(resposta.headers["Content-Length"])


def baixar(base, caminho_remoto):
    url = f"{URL_BASE}/{caminho_remoto}"
    destino = PASTA_RAW / base / Path(caminho_remoto).name
    destino.parent.mkdir(parents=True, exist_ok=True)
    total = tamanho_remoto(url)

    for tentativa in range(1, 21):
        ja_baixado = destino.stat().st_size if destino.exists() else 0
        if ja_baixado == total:
            print(f"ok       {destino.name} ({total / 1e6:.0f} MB)", flush=True)
            return destino.name, True
        if ja_baixado > total:  # arquivo local corrompido: recomeça
            destino.unlink()
            ja_baixado = 0
        try:
            pedido = urllib.request.Request(url, headers={"Range": f"bytes={ja_baixado}-"})
            with urllib.request.urlopen(pedido, timeout=120) as resposta, open(destino, "ab") as saida:
                proximo_aviso = time.time() + 60
                while bloco := resposta.read(BLOCO):
                    saida.write(bloco)
                    ja_baixado += len(bloco)
                    if time.time() > proximo_aviso:
                        print(f"baixando {destino.name}: {100 * ja_baixado / total:.0f}%", flush=True)
                        proximo_aviso = time.time() + 60
        except Exception as erro:  # rede caiu: espera e retoma do ponto em que parou
            print(f"falha    {destino.name} (tentativa {tentativa}): {erro}", flush=True)
            time.sleep(min(60, 5 * tentativa))

    return destino.name, False


def main():
    bases = sys.argv[1:] or list(ARQUIVOS)
    desconhecidas = [b for b in bases if b not in ARQUIVOS]
    if desconhecidas:
        sys.exit(f"Base desconhecida: {desconhecidas}. Opções: {list(ARQUIVOS)}")

    tarefas = [(base, arquivo) for base in bases for arquivo in ARQUIVOS[base]]
    with ThreadPoolExecutor(max_workers=4) as executor:
        resultados = list(executor.map(lambda t: baixar(*t), tarefas))

    faltando = [nome for nome, ok in resultados if not ok]
    if faltando:
        sys.exit(f"NÃO baixados: {faltando}")
    print(f"Tudo baixado em {PASTA_RAW}")


if __name__ == "__main__":
    main()

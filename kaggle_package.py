"""Monta as duas pastas a subir como datasets privados no Kaggle.

    lesoes-pele-meta/   labels.csv, md5.csv, o código (prepare_data.py, pipeline.py) e o CSV
                        com os diagnósticos de dermatologistas no teste (Tschandl et al., 2020)
    lesoes-pele-extra/  imagens de test/ e unlabeled/, que não têm cópia confiável no Kaggle

As imagens de treino/validação vêm do dataset público andrewmvd/isic-2019; o md5.csv permite
ao notebook do Kaggle conferir que são os mesmos arquivos usados aqui.

Uso:
    python kaggle_package.py USUARIO_KAGGLE [DESTINO]
DESTINO padrão: ~/datasets/kaggle. As imagens entram como hard links (não ocupam espaço extra).
"""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

from prepare_data import DEFAULT_DIR

REPO = Path(__file__).resolve().parent
CODE = ["prepare_data.py", "pipeline.py"]
# diagnósticos de dermatologistas no teste do ISIC 2018 (Tschandl et al., 2020), para comparação
READERS_CSV = Path(os.environ.get(
    "READERS_CSV",
    REPO.parent / "dataverse_files" / "ISIC2018_Task3_Test_NatureMedicine_AI_Interaction_Benefit.csv"))
DATASETS = {
    "lesoes-pele-meta": "Lesões de pele - rótulos e splits",
    "lesoes-pele-extra": "Lesões de pele - teste e sem rótulo",
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_metadata(folder: Path, user: str, title: str) -> None:
    meta = {"title": title, "id": f"{user}/{folder.name}", "licenses": [{"name": "CC-BY-NC-4.0"}]}
    (folder / "dataset-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:  # outro disco ou sistema sem hard link
        shutil.copyfile(src, dst)


def package(user: str, out_dir: Path, data_dir: Path = DEFAULT_DIR) -> None:
    df = pd.read_csv(data_dir / "labels.csv", low_memory=False)

    meta = out_dir / "lesoes-pele-meta"
    meta.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(data_dir / "labels.csv", meta / "labels.csv")
    for name in CODE:
        shutil.copyfile(REPO / name, meta / name)
    if READERS_CSV.exists():
        shutil.copyfile(READERS_CSV, meta / READERS_CSV.name)
    else:
        print(f"Aviso: {READERS_CSV} não encontrado; a comparação com dermatologistas fica de fora")
    md5_file = meta / "md5.csv"
    if not md5_file.exists():
        print(f"Calculando md5 de {len(df)} imagens...")
        pd.DataFrame({"file_id": df["file_id"],
                      "md5": [md5(data_dir / p) for p in df["image_path"]]}).to_csv(md5_file, index=False)
    write_metadata(meta, user, DATASETS[meta.name])

    extra = out_dir / "lesoes-pele-extra"
    part = df[df["split"].isin(["test", "unlabeled"])]
    print(f"Ligando {len(part)} imagens em {extra}...")
    for rel in part["image_path"]:
        link(data_dir / rel, extra / rel)
    write_metadata(extra, user, DATASETS[extra.name])


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / "datasets" / "kaggle"
    package(sys.argv[1], out_dir)
    print(f"Pronto: {out_dir}")


if __name__ == "__main__":
    main()

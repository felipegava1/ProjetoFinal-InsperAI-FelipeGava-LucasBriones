"""Organiza o HAM10000 / ISIC2018 Task 3 em pastas por classe, prontas para treino.

Estrutura gerada em DESTINO (padrão: ~/datasets/ham10000):
    train/<dx>/ISIC_xxxxxxx.jpg   HAM10000 menos a validação
    val/<dx>/ISIC_xxxxxxx.jpg     ~15% das lesões do HAM10000, estratificado por dx
    test/<dx>/ISIC_xxxxxxx.jpg    ISIC2018 Task 3 Test
    masks/ISIC_xxxxxxx_segmentation.png   (só train/val)
    labels.csv                    metadados + split + caminhos relativos

Uso:
    python organize_dataset.py ORIGEM [DESTINO]
ORIGEM é o dataverse_files.zip baixado ou a pasta onde ele foi extraído.
"""
import os
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

DEFAULT_DIR = Path(os.environ.get("HAM10000_DIR", Path.home() / "datasets" / "ham10000"))
VAL_FRAC = 0.15
SEED = 42

DX_NAMES = {
    "nv": "Nevo melanocítico",
    "mel": "Melanoma",
    "bkl": "Lesão benigna tipo queratose",
    "bcc": "Carcinoma basocelular",
    "akiec": "Queratose actínica / carcinoma intraepitelial",
    "vasc": "Lesão vascular",
    "df": "Dermatofibroma",
}
MALIGNANT = {"mel", "bcc", "akiec"}

TRAIN_ZIPS = ["HAM10000_images_part_1.zip", "HAM10000_images_part_2.zip"]
TEST_ZIP = "ISIC2018_Task3_Test_Images.zip"
MASKS_ZIP = "HAM10000_segmentations_lesion_tschandl.zip"


def split_train_val(meta: pd.DataFrame) -> pd.Series:
    # Sorteia lesões, não imagens: fotos da mesma lesão ficam no mesmo split
    lesions = meta.drop_duplicates("lesion_id")[["lesion_id", "dx"]]
    val_lesions = lesions.groupby("dx").sample(frac=VAL_FRAC, random_state=SEED)["lesion_id"]
    return meta["lesion_id"].isin(val_lesions).map({True: "val", False: "train"})


def extract_into(zip_path: Path, dest_for: dict, out_dir: Path) -> None:
    """Extrai de zip_path os arquivos cujo nome está em dest_for, cada um no seu destino."""
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if info.is_dir() or "__MACOSX" in info.filename or name not in dest_for:
                continue
            target = out_dir / dest_for[name]
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def organize(src_dir: Path, out_dir: Path) -> pd.DataFrame:
    train = pd.read_csv(src_dir / "HAM10000_metadata")
    train["split"] = split_train_val(train)
    test = pd.read_csv(src_dir / "ISIC2018_Task3_Test_GroundTruth.csv")
    test["split"] = "test"

    df = pd.concat([train, test], ignore_index=True)
    df["dx_name"] = df["dx"].map(DX_NAMES)
    df["malignant"] = df["dx"].isin(MALIGNANT).astype(int)
    df["image_path"] = df["split"] + "/" + df["dx"] + "/" + df["image_id"] + ".jpg"
    df["mask_path"] = ("masks/" + df["image_id"] + "_segmentation.png").where(df["split"] != "test")

    images = dict(zip(df["image_id"] + ".jpg", df["image_path"]))
    for z in TRAIN_ZIPS + [TEST_ZIP]:
        print(f"Extraindo {z}...")
        extract_into(src_dir / z, images, out_dir)
    masks = {Path(p).name: p for p in df["mask_path"].dropna()}
    print(f"Extraindo {MASKS_ZIP}...")
    extract_into(src_dir / MASKS_ZIP, masks, out_dir)

    # ISIC_0035068 está no GroundTruth de teste mas não vem no zip de imagens
    exists = df["image_path"].map(lambda p: (out_dir / p).exists())
    if not exists.all():
        print(f"Aviso: descartando {(~exists).sum()} imagem(ns) sem arquivo: "
              f"{df.loc[~exists, 'image_id'].tolist()}")
    df = df[exists].reset_index(drop=True)
    df.to_csv(out_dir / "labels.csv", index=False)
    return df


def load_labels(data_dir: Path = DEFAULT_DIR) -> pd.DataFrame:
    """Lê o labels.csv e devolve os caminhos já absolutos."""
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "labels.csv")
    for col in ("image_path", "mask_path"):
        df[col] = df[col].map(lambda p: str(data_dir / p) if isinstance(p, str) else p)
    return df


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp = None
    if src.suffix == ".zip":
        # o zip do Dataverse contém outros zips; desempacota só o primeiro nível
        tmp = out_dir / "_raw"
        print(f"Desempacotando {src.name} em {tmp}...")
        with zipfile.ZipFile(src) as zf:
            zf.extractall(tmp)
        src = tmp

    df = organize(src, out_dir)
    if tmp is not None:
        shutil.rmtree(tmp)

    print(f"\n{len(df)} imagens organizadas em {out_dir}")
    print(pd.crosstab(df["dx"], df["split"], margins=True))


if __name__ == "__main__":
    main()

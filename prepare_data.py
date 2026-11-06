"""Monta a base única de lesões de pele a partir do HAM10000 e do ISIC 2019.

O treino do ISIC 2019 já contém o HAM10000 inteiro (mesmos arquivos) mais BCN_20000 e MSK,
então ele é a base de treino. Do HAM10000 vêm o teste rotulado (ISIC 2018 Task 3), as máscaras
e os metadados extras (dx_type, subconjunto) das imagens do HAM.

Estrutura gerada em DESTINO (padrão: ~/datasets/lesoes_pele):
    train/<dx>/ISIC_xxxxxxx.jpg   ~85% das lesões do ISIC 2019, estratificado por dx (8 classes)
    val/<dx>/ISIC_xxxxxxx.jpg     ~15% das lesões
    test/ISIC_xxxxxxx.jpg         teste do ISIC 2018; rótulo em labels.csv (dx_ham, 7 classes)
    unlabeled/ISIC_xxxxxxx.jpg    resto do teste do ISIC 2019, sem rótulo
    masks/ISIC_xxxxxxx_segmentation.png   só imagens do HAM10000
    labels.csv                    uma linha por imagem, caminhos relativos a DESTINO

Uso:
    python prepare_data.py PASTA_HAM PASTA_ISIC2019 [DESTINO]
PASTA_HAM é o dataverse_files.zip ou a pasta onde ele foi extraído.
PASTA_ISIC2019 tem os CSVs do ISIC 2019 e as imagens (extraídas ou nos zips).
"""
import os
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

DEFAULT_DIR = Path(os.environ.get("LESOES_DIR", Path.home() / "datasets" / "lesoes_pele"))
VAL_FRAC = 0.15
SEED = 42

CLASSES = ["ak", "bcc", "bkl", "df", "mel", "nv", "scc", "vasc"]
HAM_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
TO_HAM = {"ak": "akiec", "scc": "akiec"}  # o HAM10000 não separa ak de scc

DX_NAMES = {
    "nv": "Nevo melanocítico",
    "mel": "Melanoma",
    "bcc": "Carcinoma basocelular",
    "scc": "Carcinoma espinocelular",
    "ak": "Queratose actínica",
    "akiec": "Queratose actínica / carcinoma espinocelular",
    "bkl": "Lesão benigna tipo queratose",
    "vasc": "Lesão vascular",
    "df": "Dermatofibroma",
}
# ak entra como maligna para manter a mesma definição do akiec no HAM10000
MALIGNANT = {"mel", "bcc", "scc", "ak", "akiec"}

# localization (HAM) -> anatom_site_general (ISIC 2019); trunk não diz se é anterior/posterior
HAM_SITES = {
    "abdomen": "anterior torso", "chest": "anterior torso", "back": "posterior torso",
    "face": "head/neck", "ear": "head/neck", "neck": "head/neck", "scalp": "head/neck",
    "upper extremity": "upper extremity", "hand": "upper extremity",
    "lower extremity": "lower extremity", "foot": "lower extremity",
    "acral": "palms/soles", "genital": "oral/genital",
}


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


def copy_isic2019(src_dir: Path, folder: str, dest_for: dict, out_dir: Path) -> None:
    """Copia da pasta extraída do ISIC 2019, ou direto do zip se ela não existir."""
    extracted = next((p for p in (src_dir / folder / folder, src_dir / folder) if p.is_dir()), None)
    if extracted is None:
        extract_into(src_dir / f"{folder}.zip", dest_for, out_dir)
        return
    for name, rel in dest_for.items():
        target = out_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(extracted / name, target)


def split_train_val(df: pd.DataFrame) -> pd.Series:
    # Sorteia lesões, não imagens; imagens sem lesion_id contam como lesão própria
    group = df["lesion_id"].fillna(df["image_id"])
    lesions = pd.DataFrame({"group": group, "dx": df["dx"]}).drop_duplicates("group")
    val = lesions.groupby("dx").sample(frac=VAL_FRAC, random_state=SEED)["group"]
    return group.isin(val).map({True: "val", False: "train"})


def build_table(ham_dir: Path, isic_dir: Path) -> pd.DataFrame:
    gt = pd.read_csv(isic_dir / "ISIC_2019_Training_GroundTruth.csv")
    train = pd.read_csv(isic_dir / "ISIC_2019_Training_Metadata.csv").merge(gt, on="image")
    train["dx"] = train[gt.columns[1:]].idxmax(axis=1).str.lower()
    train = train.drop(columns=gt.columns[1:]).rename(columns={"image": "file_id"})
    # imagens do MSK têm sufixo _downsampled no nome do arquivo
    train["image_id"] = train["file_id"].str.extract(r"(ISIC_\d+)")[0]
    train["split"] = split_train_val(train)
    train["dx_ham"] = train["dx"].replace(TO_HAM)

    test = pd.read_csv(ham_dir / "ISIC2018_Task3_Test_GroundTruth.csv")
    test = test.rename(columns={"dx": "dx_ham", "age": "age_approx"})
    test["file_id"] = test["image_id"]
    test["split"] = "test"

    unlabeled = pd.read_csv(isic_dir / "ISIC_2019_Test_Metadata.csv").rename(columns={"image": "file_id"})
    unlabeled["image_id"] = unlabeled["file_id"]
    # o teste do ISIC 2019 contém o teste do ISIC 2018 inteiro
    unlabeled = unlabeled[~unlabeled["image_id"].isin(test["image_id"])]
    unlabeled["split"] = "unlabeled"

    df = pd.concat([train, test, unlabeled], ignore_index=True)

    ham = pd.read_csv(ham_dir / "HAM10000_metadata")
    is_ham = df["image_id"].isin(ham["image_id"]) | (df["split"] == "test")
    df["source"] = df["lesion_id"].str.extract(r"^(BCN|MSK)")[0]
    df.loc[df["split"].isin(["train", "val"]) & df["source"].isna(), "source"] = "MSK"
    df.loc[is_ham, "source"] = "HAM"

    extra = ham.set_index("image_id")[["dx_type", "dataset", "localization"]]
    df = df.join(extra, on="image_id", rsuffix="_ham")
    for col in ("dx_type", "dataset", "localization"):
        df[col] = df[col].fillna(df.pop(f"{col}_ham"))
    df["site"] = df["anatom_site_general"].fillna(df["localization"].map(HAM_SITES))
    df["sex"] = df["sex"].replace("unknown", pd.NA)

    df["dx_name"] = df["dx"].fillna(df["dx_ham"]).map(DX_NAMES)
    df["malignant"] = df["dx_ham"].isin(MALIGNANT).astype(int).where(df["dx_ham"].notna())

    folder = df["split"] + "/" + df["dx"].where(df["split"].isin(["train", "val"])).add("/").fillna("")
    df["image_path"] = folder + df["file_id"] + ".jpg"
    df["mask_path"] = ("masks/" + df["image_id"] + "_segmentation.png").where(
        df["image_id"].isin(ham["image_id"]))

    cols = ["image_id", "file_id", "lesion_id", "split", "dx", "dx_ham", "dx_name", "malignant",
            "source", "dataset", "dx_type", "age_approx", "sex", "site", "image_path", "mask_path"]
    return df[cols]


def prepare(ham_dir: Path, isic_dir: Path, out_dir: Path) -> pd.DataFrame:
    df = build_table(ham_dir, isic_dir)

    for folder, splits in [("ISIC_2019_Training_Input", ["train", "val"]),
                           ("ISIC_2019_Test_Input", ["unlabeled"])]:
        part = df[df["split"].isin(splits)]
        print(f"Copiando {len(part)} imagens de {folder}...")
        copy_isic2019(isic_dir, folder, dict(zip(part["file_id"] + ".jpg", part["image_path"])), out_dir)

    part = df[df["split"] == "test"]
    print(f"Extraindo {len(part)} imagens de teste do ISIC 2018...")
    extract_into(ham_dir / "ISIC2018_Task3_Test_Images.zip",
                 dict(zip(part["file_id"] + ".jpg", part["image_path"])), out_dir)
    masks = df["mask_path"].dropna()
    print(f"Extraindo {len(masks)} máscaras...")
    extract_into(ham_dir / "HAM10000_segmentations_lesion_tschandl.zip",
                 {Path(p).name: p for p in masks}, out_dir)

    # ISIC_0035068 está no gabarito do ISIC 2018 mas não vem no zip de imagens
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
    df = pd.read_csv(data_dir / "labels.csv", low_memory=False)
    for col in ("image_path", "mask_path"):
        df[col] = df[col].map(lambda p: str(data_dir / p) if isinstance(p, str) else p)
    return df


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    ham_src, isic_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp = None
    if ham_src.suffix == ".zip":
        # o zip do Dataverse contém outros zips; desempacota só o primeiro nível
        tmp = out_dir / "_raw"
        print(f"Desempacotando {ham_src.name} em {tmp}...")
        with zipfile.ZipFile(ham_src) as zf:
            zf.extractall(tmp)
        ham_src = tmp

    df = prepare(ham_src, isic_dir, out_dir)
    if tmp is not None:
        shutil.rmtree(tmp)

    print(f"\n{len(df)} imagens organizadas em {out_dir}")
    print(pd.crosstab(df["dx"].fillna(df["dx_ham"]).fillna("(sem rótulo)"), df["split"], margins=True))


if __name__ == "__main__":
    main()

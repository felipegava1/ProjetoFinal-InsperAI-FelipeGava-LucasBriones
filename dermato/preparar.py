"""Monta a divisão fixa splits/dermato.csv e o cache de imagens (menor lado 448 px).

Uso:
    python -m dermato.preparar splits     # gera splits/dermato.csv (só CSVs; rápido)
    python -m dermato.preparar cache      # redimensiona as imagens que faltam para DATA/cache_448
    python -m dermato.preparar tudo

Fontes (só dermatoscopia, 8 classes ak bcc bkl df mel nv scc vasc):
    treino/val  ISIC 2019 treino; metade A do teste oficial do ISIC 2019 (sem as imagens do ISIC 2018);
                ISIC 2020 só com diagnóstico específico (melanoma, nevo, ceratoses benignas, lentigos);
                MILK10k só a parte dermatoscópica.
    t1          ISIC 2018 Task 3 teste (1.511 imagens, rótulo em 7 classes HAM, coluna dx_ham)
    t2          metade B do teste oficial do ISIC 2019, sem as imagens do ISIC 2018
A validação é 15% do conjunto de treino, sorteada por grupo (paciente ou lesão; imagem quando a
base não informa), estratificada por classe, semente 42. O arquivo é gerado uma vez e não muda.
"""
import io
import sys
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .config import CACHE, CACHE_LADO, CLASSES, EXTRAIDO, PARA_HAM, RAW, SEED, SITES, SPLITS

PARA_SITE = {s: s for s in SITES} | {
    "head_neck_face": "head/neck", "upper_extremity": "upper extremity", "hand": "upper extremity",
    "lower_extremity": "lower extremity", "foot": "lower extremity", "genital": "oral/genital",
    "palms/soles": "palms/soles", "oral/genital": "oral/genital",
}   # "torso"/"trunk" sem frente/costas fica faltante, como no HAM10000

ISIC2019 = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
CLASSE_2020 = {"melanoma": "mel", "nevus": "nv", "seborrheic keratosis": "bkl",
               "lichenoid keratosis": "bkl", "solar lentigo": "bkl", "lentigo NOS": "bkl"}


def onehot(df, cols):
    return df[cols].idxmax(axis=1).where(df[cols].sum(axis=1) == 1)


def indice_imagens():
    """isic_id -> caminho do original, procurando em todas as pastas extraídas."""
    achados = {}
    for p in EXTRAIDO.rglob("*.jp*g"):
        achados.setdefault(p.stem, str(p))
    return achados


def ler_isic2018_teste():
    with zipfile.ZipFile(RAW / "ISIC2018_Task3_Test_GroundTruth.zip") as z:
        nome = next(n for n in z.namelist() if n.endswith(".csv"))
        gt = pd.read_csv(z.open(nome))
    cols = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    return pd.DataFrame({"isic_id": gt["image"], "dx_ham": onehot(gt, cols).str.lower()})


PARES = SPLITS.parent / "pares_duplicatas.csv.gz"     # gerado por python -m dermato.duplicatas pares
REMOVIDAS = SPLITS.parent / "removidas_duplicatas.csv"
LIM_LESAO = 0.92     # cosseno do PanDerm para "mesma lesão" ao estender a remoção ligada ao T1
# todo par do arquivo (cos do PanDerm >= 0,90 ou mesma foto pelo ORB) junta as duas imagens no mesmo grupo


def _componentes(n, a, b, grupos):
    """Componentes conexos de um grafo com arestas (a, b) mais 'mesmo valor em grupos'."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    g = pd.factorize(grupos)[0]
    lin = np.r_[a, np.arange(n)]
    col = np.r_[b, n + g]
    A = coo_matrix((np.ones(len(lin)), (lin, col)), shape=(n + g.max() + 1,) * 2)
    return connected_components(A, directed=False)[1][:n]


def aplicar_duplicatas(df):
    """Divisão v2 (14/11): agrupa imagens quase idênticas para que a mesma lesão não caia em dois splits.

    1. T1 (ISIC 2018 teste) fica inteiro, para a comparação com os dermatologistas. Sai do conjunto toda
       imagem de outra base que forma par com uma imagem do T1 e as fotos da mesma lesão que ela
       (mesmo lesion_id, cos >= 0,92 ou mesma foto pelo ORB, encadeado).
    2. O grupo de cada imagem passa a ser o componente conexo dos pares unido aos grupos originais
       (paciente/lesão). O sorteio de T2 e da validação usa esse grupo.
    """
    if not PARES.exists():
        raise FileNotFoundError(f"{PARES} não existe: rode python -m dermato.duplicatas pares")
    df = df.reset_index(drop=True)
    n = len(df)
    pos = pd.Series(np.arange(n), index=df["isic_id"])
    p = pd.read_csv(PARES)
    p = p[p["a"].isin(pos.index) & p["b"].isin(pos.index)]
    a, b, c = pos[p["a"]].values, pos[p["b"]].values, p["cos"].values
    t1 = (df["split"] == "t1").to_numpy()

    # 1. remoção ligada ao T1
    lesao = df["grupo"].where(~df["grupo"].str.startswith("isic2020:"), df["isic_id"])   # 2020 é paciente
    forte = (c >= LIM_LESAO) | (p["criterio"].values == "orb")
    comp_lesao = _componentes(n, a[forte], b[forte], lesao)
    vizinho = np.zeros(n, bool)
    lig = t1[a] ^ t1[b]                                    # pares T1 x não-T1
    vizinho[np.where(t1[a], b, a)[lig]] = True
    marcados = np.isin(comp_lesao, np.unique(comp_lesao[t1 | vizinho]))
    sai = marcados & ~t1
    removidas = df.loc[sai, ["isic_id", "base", "dx", "grupo"]].assign(
        motivo=np.where(vizinho[sai], "par com imagem do T1", "mesma lesão de um par do T1"))
    df = df.loc[~sai].reset_index(drop=True)

    # 2. grupos unidos
    n = len(df)
    pos = pd.Series(np.arange(n), index=df["isic_id"])
    p = p[p["a"].isin(pos.index) & p["b"].isin(pos.index)]
    comp = _componentes(n, pos[p["a"]].values, pos[p["b"]].values, df["grupo"])
    df["grupo"] = pd.Series(comp).map(lambda k: f"g{k:05d}").values
    return df, removidas.reset_index(drop=True)


def montar_splits():
    linhas = []
    # ---- ISIC 2019 treino
    gt = pd.read_csv(RAW / "ISIC_2019_Training_GroundTruth.csv")
    meta = pd.read_csv(RAW / "ISIC_2019_Training_Metadata.csv")
    d = gt.assign(dx=onehot(gt, ISIC2019).str.lower())[["image", "dx"]].merge(meta, on="image")
    linhas.append(pd.DataFrame({
        "isic_id": d["image"], "base": "isic2019_train", "dx": d["dx"],
        "grupo": "isic2019:" + d["lesion_id"].fillna(d["image"]),
        "idade": d["age_approx"], "sexo": d["sex"], "local": d["anatom_site_general"]}))

    # ---- ISIC 2018 teste (T1) e ISIC 2019 teste oficial (metade treino, metade T2)
    t1 = ler_isic2018_teste()
    t1 = t1[t1["isic_id"] != "ISIC_0035068"]          # está no gabarito, mas não no zip
    gt = pd.read_csv(RAW / "ISIC_2019_Test_GroundTruth.csv")
    meta = pd.read_csv(RAW / "ISIC_2019_Test_Metadata.csv")
    d = gt.assign(dx=onehot(gt, ISIC2019).str.lower())[["image", "dx"]].merge(meta, on="image")
    meta_t1 = d.set_index("image")
    t1 = t1.assign(dx=meta_t1["dx"].reindex(t1["isic_id"]).values)
    linhas.append(pd.DataFrame({
        "isic_id": t1["isic_id"], "base": "isic2018_test", "dx": t1["dx"], "dx_ham": t1["dx_ham"],
        "grupo": "isic2018:" + t1["isic_id"], "split": "t1",
        "idade": meta_t1["age_approx"].reindex(t1["isic_id"]).values,
        "sexo": meta_t1["sex"].reindex(t1["isic_id"]).values,
        "local": meta_t1["anatom_site_general"].reindex(t1["isic_id"]).values}))
    d = d[~d["image"].isin(t1["isic_id"]) & d["dx"].isin(CLASSES)].reset_index(drop=True)
    linhas.append(pd.DataFrame({                              # metades A/B decididas por grupo, mais abaixo
        "isic_id": d["image"], "base": "isic2019_test",
        "dx": d["dx"], "grupo": "isic2019test:" + d["image"],   # o teste não informa lesion_id
        "idade": d["age_approx"], "sexo": d["sex"], "local": d["anatom_site_general"]}))

    # ---- ISIC 2020 (só diagnóstico específico, sem as duplicatas declaradas)
    gt = pd.read_csv(RAW / "ISIC_2020_Training_GroundTruth_v2.csv")
    dup = pd.read_csv(RAW / "ISIC_2020_Training_Duplicates.csv")
    gt = gt[~gt["image_name"].isin(dup["image_name_2"])]
    gt = gt.assign(dx=gt["diagnosis"].map(CLASSE_2020)).dropna(subset=["dx"])
    linhas.append(pd.DataFrame({
        "isic_id": gt["image_name"], "base": "isic2020", "dx": gt["dx"],
        "grupo": "isic2020:" + gt["patient_id"],
        "idade": gt["age_approx"], "sexo": gt["sex"], "local": gt["anatom_site_general_challenge"]}))

    # ---- MILK10k, só dermatoscopia
    inp = pd.read_csv(RAW / "milk10k_training_input.csv")
    sup = pd.read_csv(RAW / "milk10k_training_supp.csv")
    gt = pd.read_csv(RAW / "milk10k_training_gt.csv")
    cols = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
    gt["rotulo"] = onehot(gt, cols)
    d = inp[inp["image_type"] == "dermoscopic"].merge(sup, on="isic_id").merge(gt[["lesion_id", "rotulo"]], on="lesion_id")
    akiec = np.where(d["diagnosis_full"] == "Solar or actinic keratosis", "ak", "scc")
    d["dx"] = np.where(d["rotulo"] == "AKIEC", akiec,
                       d["rotulo"].map({"BCC": "bcc", "BKL": "bkl", "DF": "df", "MEL": "mel",
                                        "NV": "nv", "SCCKA": "scc", "VASC": "vasc"}))
    d = d[d["dx"].isin(CLASSES)]
    linhas.append(pd.DataFrame({
        "isic_id": d["isic_id"], "base": "milk10k", "dx": d["dx"], "grupo": "milk10k:" + d["lesion_id"],
        "idade": d["age_approx"], "sexo": d["sex"], "local": d["site"]}))

    df = pd.concat(linhas, ignore_index=True)
    assert df["isic_id"].is_unique, "isic_id repetido entre bases"
    df["local"] = df["local"].map(PARA_SITE)
    df["sexo"] = df["sexo"].where(df["sexo"].isin(["male", "female"]))
    df.loc[df["dx_ham"].isna() & df["dx"].notna(), "dx_ham"] = df["dx"].map(PARA_HAM)

    # ---- imagens quase idênticas (mesma foto ou mesma lesão com outro ID): ver dermato/duplicatas.py
    df, removidas = aplicar_duplicatas(df)

    # ---- teste oficial do ISIC 2019: metade A (treino) e B (T2), sorteadas por grupo. Grupos que também
    #      têm imagens de outras bases ficam na metade A, para o T2 não ter nenhuma lesão vista no treino.
    t19 = (df["base"] == "isic2019_test").to_numpy()
    so_teste = df.groupby("grupo")["base"].transform(lambda b: (b == "isic2019_test").all()).to_numpy()
    metade = np.zeros(len(df), dtype=int)
    cand = np.where(t19 & so_teste)[0]
    sub = df.iloc[cand]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, idx_b = next(StratifiedGroupKFold(2, shuffle=True, random_state=SEED).split(sub, sub["dx"], sub["grupo"]))
    metade[cand[idx_b]] = 1
    df.loc[t19, "base"] = np.where(metade[t19] == 1, "isic2019_test_b", "isic2019_test_a")
    df.loc[t19 & (metade == 1), "split"] = "t2"

    # ---- validação: 15% (3 de 20 dobras) do conjunto de treino, por grupo
    # grupos maiores que meia dobra (o encadeamento de duplicatas + pacientes do ISIC 2020 forma um grupo
    # de ~3.700 imagens) vão direto para o treino: na validação distorceriam a composição dela
    tam = df.groupby("grupo")["isic_id"].transform("size")
    limite = df["split"].isna().sum() / 40
    df.loc[df["split"].isna() & (tam > limite), "split"] = "train"
    pool = df["split"].isna().to_numpy()
    sgkf = StratifiedGroupKFold(20, shuffle=True, random_state=SEED)
    dobra = np.zeros(pool.sum(), dtype=int)
    sub = df[pool]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, (_, idx) in enumerate(sgkf.split(sub, sub["dx"], sub["grupo"])):
            dobra[idx] = i
    df.loc[pool, "split"] = np.where(dobra < 3, "val", "train")

    # sem vazamento de grupo entre treino, validação e T2 (T1 é por imagem; tratado em aplicar_duplicatas)
    g = df[df["split"].isin(["train", "val", "t2"])].groupby("grupo")["split"].nunique()
    assert (g == 1).all(), "grupo dividido entre splits"
    # nenhum par quase idêntico entre splits diferentes (exceto T1 x T2, que nunca treinam)
    if PARES.exists():
        p = pd.read_csv(PARES)
        sp = df.set_index("isic_id")["split"]
        a, b = sp.reindex(p["a"]).values, sp.reindex(p["b"]).values
        ok = pd.notna(a) & pd.notna(b)
        cruz = ok & (a != b) & ~(np.isin(a, ["t1", "t2"]) & np.isin(b, ["t1", "t2"]))
        assert not cruz.any(), f"{cruz.sum()} pares quase idênticos entre splits"

    cols = ["isic_id", "base", "split", "dx", "dx_ham", "grupo", "idade", "sexo", "local"]
    df = df[cols].sort_values(["split", "base", "isic_id"]).reset_index(drop=True)
    SPLITS.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SPLITS, index=False)
    removidas.to_csv(REMOVIDAS, index=False)
    print(f"removidas por duplicata do T1: {len(removidas)} -> {REMOVIDAS}")
    pd.set_option("display.width", 200)
    print(pd.crosstab(df["base"], df["split"], margins=True))
    print(pd.crosstab(df["dx"], df["split"], margins=True))
    print(f"-> {SPLITS} ({len(df)} imagens)")
    return df


def _redimensionar(args):
    origem, destino = args
    try:
        im = Image.open(origem)
        menor = min(im.size)
        escala = min(1.0, CACHE_LADO / menor)
        novo = (max(1, round(im.width * escala)), max(1, round(im.height * escala)))
        if im.format == "JPEG":
            im.draft("RGB", novo)
        im = im.convert("RGB")
        if im.size != novo:
            im = im.resize(novo, Image.Resampling.LANCZOS)
        tmp = destino + ".tmp"
        im.save(tmp, "JPEG", quality=95)
        Path(tmp).replace(destino)
        return ""
    except Exception as e:  # noqa: BLE001
        return f"{origem}: {e!r}"


def montar_cache():
    df = pd.read_csv(SPLITS)
    CACHE.mkdir(parents=True, exist_ok=True)
    idx = indice_imagens()
    faltando = [i for i in df["isic_id"] if i not in idx]
    if faltando:
        print(f"ATENÇÃO: {len(faltando)} imagens sem original extraído (ex.: {faltando[:5]})")
    todo = [(idx[i], str(CACHE / f"{i}.jpg")) for i in df["isic_id"]
            if i in idx and not (CACHE / f"{i}.jpg").exists()]
    print(f"Redimensionando {len(todo)} imagens para {CACHE} ...", flush=True)
    erros = []
    with ProcessPoolExecutor(16) as ex:
        for k, e in enumerate(ex.map(_redimensionar, todo, chunksize=32)):
            if e:
                erros.append(e)
            if k % 2000 == 0:
                print(f"  {k}/{len(todo)}", flush=True)
    print(f"Pronto. Erros: {len(erros)}", *erros[:10], sep="\n")
    return len(faltando) + len(erros)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tudo"
    if cmd in ("splits", "tudo"):
        montar_splits()
    if cmd in ("cache", "tudo"):
        sys.exit(1 if montar_cache() else 0)

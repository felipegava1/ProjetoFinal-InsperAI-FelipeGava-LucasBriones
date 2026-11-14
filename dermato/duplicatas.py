"""Checagem de imagens quase idênticas entre treino/validação e os testes T1/T2.

A divisão garante que nenhum isic_id e nenhum grupo (paciente/lesão) cai em dois splits, mas só onde a
base informa o grupo. O teste do ISIC 2019 não tem lesion_id e as bases se sobrepõem (MILK10k, ISIC 2020),
então a mesma lesão pode aparecer com IDs diferentes. Aqui, para cada imagem de T1, T2 e validação,
procura-se o vizinho mais próximo no treino (para T1/T2: treino + validação):

  cos   cosseno entre os vetores do PanDerm (centralizados pela média; sem centralizar, tudo dá > 0,9)
  corr  correlação de pixels do centro da imagem (48x48 em cinza, melhor das 8 orientações), que separa
        "mesma foto" de "mesma lesão, outra foto"

Testado e descartado: hash perceptual (pHash 8x8). Fotos de dermatoscopia se parecem demais (mancha no
centro, vinheta escura) e o hash marca muitos pares de lesões diferentes.

Saídas (versionadas): resultados/duplicatas.csv e resultados/figuras/duplicatas_<split>_<faixa>.jpg.
Precisa do cache de imagens e de resultados/panderm_emb.npz (python -m dermato.panderm extrair).

  python -m dermato.duplicatas          # vizinhos dos testes (checagem) + figuras
  python -m dermato.duplicatas pares    # pares quase idênticos do conjunto inteiro, usados pela divisão v2
"""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import cv2

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from .config import CACHE, REPO, SPLITS, achar

SAIDA = REPO / "resultados" / "duplicatas.csv"
FIGURAS = REPO / "resultados" / "figuras"
FAIXAS = [(0.95, 1.01), (0.90, 0.95), (0.85, 0.90), (0.80, 0.85)]


def _centro(isic_id):
    im = Image.open(CACHE / f"{isic_id}.jpg").convert("L")
    im.draft("L", (128, 128))
    return np.asarray(im.resize((96, 96), Image.BILINEAR), np.float32)[24:72, 24:72]


def _corr(par):
    a, b = _centro(par[0]), _centro(par[1])
    a = (a - a.mean()) / (a.std() + 1e-6)
    melhor = -1.0
    for k in range(4):
        for espelho in (False, True):
            c = np.rot90(b, k)
            c = c[:, ::-1] if espelho else c
            c = (c - c.mean()) / (c.std() + 1e-6)
            melhor = max(melhor, float((a * c).mean()))
    return melhor


def vizinhos():
    d = pd.read_csv(SPLITS)
    z = np.load(achar("panderm_emb.npz", ""), allow_pickle=True)
    E = pd.DataFrame(z["emb"], index=z["isic_id"]).loc[d["isic_id"]].values
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    E = torch.nn.functional.normalize(torch.tensor(E - z["emb"].astype(np.float32).mean(0), dtype=torch.float32, device=dev), dim=1)
    partes = []
    for split, refs in [("t1", ["train", "val"]), ("t2", ["train", "val"]), ("val", ["train"])]:
        te = np.where(d["split"] == split)[0]
        ref = np.where(d["split"].isin(refs))[0]
        v, j = (E[te] @ E[ref].T).max(1)
        j = ref[j.cpu().numpy()]
        partes.append(pd.DataFrame({
            "split": split, "isic_id": d["isic_id"].values[te], "base": d["base"].values[te],
            "dx": d["dx"].values[te], "viz_id": d["isic_id"].values[j], "viz_base": d["base"].values[j],
            "viz_split": d["split"].values[j], "viz_dx": d["dx"].values[j],
            "cos": np.round(v.cpu().numpy(), 4)}))
    r = pd.concat(partes, ignore_index=True)
    with ProcessPoolExecutor(8) as ex:
        r["corr"] = np.round(list(ex.map(_corr, zip(r["isic_id"], r["viz_id"]), chunksize=128)), 4)
    return r


def figuras(r, n=8, semente=1):
    FIGURAS.mkdir(parents=True, exist_ok=True)
    for split in ["t1", "t2"]:
        for lo, hi in FAIXAS:
            g = r[(r["split"] == split) & (r["cos"] >= lo) & (r["cos"] < hi)]
            if g.empty:
                continue
            g = g.sample(min(n, len(g)), random_state=semente)
            linhas = (len(g) + 1) // 2
            W = Image.new("RGB", (1040, linhas * 190), "white")
            for k, (_, x) in enumerate(g.iterrows()):
                X, Y = (k % 2) * 520, (k // 2) * 190
                rotulos = [(x["isic_id"], f"{split} {x['base']} {x['dx']}  cos {x['cos']:.3f}  corr {x['corr']:.2f}"),
                           (x["viz_id"], f"{x['viz_split']} {x['viz_base']} {x['viz_dx']}")]
                for m, (i, txt) in enumerate(rotulos):
                    im = Image.open(CACHE / f"{i}.jpg").convert("RGB")
                    im.thumbnail((255, 185))
                    W.paste(im, (X + m * 260, Y))
                    ImageDraw.Draw(W).text((X + m * 260 + 3, Y + 3), txt, fill="yellow")
            W.save(FIGURAS / f"duplicatas_{split}_{lo:.2f}.jpg", quality=80)


PARES = REPO / "splits" / "pares_duplicatas.csv.gz"


LIM_COS = 0.90       # cosseno do PanDerm a partir do qual o par é tratado como a mesma lesão
LIM_INLIERS = 25     # pontos ORB coincidentes (após RANSAC) a partir dos quais é a mesma foto
K_VIZ = 10           # candidatos ao ORB: os 10 vizinhos mais próximos pelo PanDerm de cada imagem...
LIM_CINZA = 0.90     # ...mais os pares com centro em cinza muito parecido (cos do PanDerm >= 0,5)
CINZA_RES, CINZA_LADO = 128, 48
_orb = None


def _cinza(isic_id):
    im = Image.open(CACHE / f"{isic_id}.jpg").convert("L")
    im.draft("L", (256, 256))
    a = np.asarray(im.resize((CINZA_RES, CINZA_RES), Image.BILINEAR), np.float32)
    o = (CINZA_RES - CINZA_LADO) // 2
    a = a[o:o + CINZA_LADO, o:o + CINZA_LADO].ravel()
    return (a - a.mean()) / (a.std() + 1e-6)


@lru_cache(maxsize=3000)
def _orb_feats(isic_id, espelho=False):
    global _orb
    if _orb is None:
        _orb = cv2.ORB_create(1000)
    im = cv2.imread(str(CACHE / f"{isic_id}.jpg"), cv2.IMREAD_GRAYSCALE)
    s = 400 / max(im.shape)
    im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    im = cv2.createCLAHE(2.0, (8, 8)).apply(im)
    if espelho:
        im = im[:, ::-1].copy()
    k, d = _orb.detectAndCompute(im, None)
    return (np.float32([p.pt for p in k]) if k else np.zeros((0, 2), np.float32)), d


def _inliers(a, b):
    """Pontos ORB coincidentes entre duas imagens, após RANSAC (homografia); melhor entre b e b espelhada."""
    pa, da = _orb_feats(a)
    melhor = 0
    for espelho in (False, True):
        pb, db = _orb_feats(b, espelho)
        if da is None or db is None or len(da) < 8 or len(db) < 8:
            continue
        m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
        g = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
        if len(g) < 8:
            continue
        _, mask = cv2.findHomography(pa[[x.queryIdx for x in g]], pb[[x.trainIdx for x in g]], cv2.RANSAC, 5.0)
        if mask is not None:
            melhor = max(melhor, int(mask.sum()))
    return melhor


def _lote_inliers(pares_):
    return [_inliers(a, b) for a, b in pares_]


def pares():
    """Pares quase idênticos do conjunto inteiro (inclusive imagens que a divisão acaba removendo).

    Critério: cosseno do PanDerm >= 0,90 (mesma lesão, inclusive outra foto) OU >= 25 pontos ORB
    coincidentes após RANSAC (mesma foto, mesmo recolorida ou recortada: o MILK10k recolore imagens do
    arquivo ISIC e aí o cosseno cai para 0,7-0,88). Calibração (14/11): duplicatas conhecidas, mediana de
    312 pontos (23 de 26 >= 25); 200 pares de lesões diferentes parecidas, mediana 0 (1 de 200 >= 25).
    Correlação de pixels sozinha foi testada e descartada: nevos lisos parecidos dão correlação > 0,95.
    Salvo em splits/ (versionado) para a divisão ser reproduzível sem os vetores do PanDerm."""
    z = np.load(achar("panderm_emb.npz", ""), allow_pickle=True)
    ids = np.asarray(z["isic_id"]).astype(str)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    E = z["emb"].astype(np.float32)
    E = torch.nn.functional.normalize(torch.tensor(E - E.mean(0), device=dev), dim=1)   # fp32: mesmo valor da checagem
    with ProcessPoolExecutor(8) as ex:
        G = torch.tensor(np.stack(list(ex.map(_cinza, ids, chunksize=256))), dtype=torch.float16, device=dev)
    I, J, C, CAND = [], [], [], []
    for s in range(0, len(E), 512):
        cs = (E[s:s + 512] @ E.T).float()
        q = G[s:s + 512].view(-1, CINZA_LADO, CINZA_LADO)
        vs = [torch.rot90(q, k, (1, 2)) for k in range(4)]
        vs += [v.flip(2) for v in vs]
        cz = torch.stack([v.reshape(len(q), -1) @ G.T for v in vs]).amax(0).float() / G.shape[1]
        topk = torch.zeros_like(cs, dtype=torch.bool)
        cs_sem_si = cs.clone()
        cs_sem_si[torch.arange(len(q)), torch.arange(s, s + len(q))] = -1
        topk.scatter_(1, cs_sem_si.topk(K_VIZ, dim=1).indices, True)
        cand = (cs >= LIM_COS) | (topk & (cs >= 0.6)) | ((cz >= LIM_CINZA) & (cs >= 0.5))
        i, j = torch.nonzero(cand, as_tuple=True)
        j_ = j.cpu().numpy(); i_ = i.cpu().numpy() + s
        m = i_ != j_
        a_, b_ = np.minimum(i_[m], j_[m]), np.maximum(i_[m], j_[m])
        I.append(a_); J.append(b_); C.append(cs[i[m], j[m]].cpu().numpy())
    I, J, C = map(np.concatenate, (I, J, C))
    p = pd.DataFrame({"a": ids[I], "b": ids[J], "cos": np.round(C, 4)}).drop_duplicates(["a", "b"])
    print(f"{len(p)} pares candidatos; ORB nos que têm cos < {LIM_COS}...")
    orb = p[p["cos"] < LIM_COS].sort_values("a")
    lista = list(zip(orb["a"], orb["b"]))
    blocos = [lista[k:k + 2000] for k in range(0, len(lista), 2000)]
    with ProcessPoolExecutor(8) as ex:
        inl = [x for r in ex.map(_lote_inliers, blocos) for x in r]
    p["inliers"] = pd.Series(inl, index=orb.index).reindex(p.index)
    p = p[(p["cos"] >= LIM_COS) | (p["inliers"] >= LIM_INLIERS)].copy()
    p["criterio"] = np.where(p["cos"] >= LIM_COS, "panderm", "orb")
    p.sort_values(["a", "b"]).to_csv(PARES, index=False)
    print(f"{len(p)} pares ({(p['criterio'] == 'orb').sum()} só pelo ORB) -> {PARES}")
    return p


def main():
    import sys
    if sys.argv[1:] == ["pares"]:
        pares()
        return
    r = vizinhos()
    r.to_csv(SAIDA, index=False)
    figuras(r)
    for s, g in r.groupby("split"):
        print(s, len(g), {t: int((g["cos"] >= t).sum()) for t in (0.95, 0.90, 0.85)})
    print(f"-> {SAIDA}")


if __name__ == "__main__":
    main()

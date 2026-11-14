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

  python -m dermato.duplicatas
"""
from concurrent.futures import ProcessPoolExecutor

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
    E = torch.nn.functional.normalize(torch.tensor(E - E.mean(0), dtype=torch.float32, device=dev), dim=1)
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


def main():
    r = vizinhos()
    r.to_csv(SAIDA, index=False)
    figuras(r)
    for s, g in r.groupby("split"):
        print(s, len(g), {t: int((g["cos"] >= t).sum()) for t in (0.95, 0.90, 0.85)})
    print(f"-> {SAIDA}")


if __name__ == "__main__":
    main()

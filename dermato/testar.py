"""Abre os testes T1 (ISIC 2018) e T2 (metade do teste ISIC 2019) — UMA vez, só nos finalistas.

Uso:
    python -m dermato.testar prever E0_final_s42 E1_final_s42 ...   # gera previsões de teste (TTA)
    python -m dermato.testar relatorio --base E0_final_s42 --finalistas E1_final_s42 E5_ensemble ...

`prever` reconstrói cada modelo a partir da config em experimentos.csv e do .pt salvo e grava
RESULTADOS/previsoes/<id>_t1.npz e _t2.npz. `relatorio` usa tau e limiar escolhidos NA VALIDAÇÃO,
calcula as métricas de teste, o bootstrap pareado por imagem contra o baseline (2.000 amostras) e
compara com os dermatologistas de Tschandl et al. (2020) nas imagens que eles avaliaram.
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from .config import CLASSES, EXPERIMENTOS, HAM_CLASSES, RAW, RESULTADOS, SEED
from .dados import ler_splits, loader
from .metricas import avaliar, escore_maligno, para_ham, ajustar
from .treinar import PADRAO, carregar_embeddings, construir, prever

PREV = RESULTADOS / "previsoes"


def cfg_de(rid):
    exp = pd.read_csv(EXPERIMENTOS)
    linha = exp[exp["id"] == rid].iloc[-1]
    return PADRAO | json.loads(linha["config"]), linha["modelo"]


def prever_testes(rids):
    df = ler_splits()
    df["linha"] = np.arange(len(df))
    for rid in rids:
        cfg, caminho = cfg_de(rid)
        model, mean, std, micro, _ = construir(cfg | {"pesos_cnn": None})
        model.load_state_dict(torch.load(caminho, map_location="cpu", weights_only=True))
        model = model.cuda().to(memory_format=torch.channels_last)
        emb = torch.from_numpy(carregar_embeddings(cfg["emb"])) if cfg["emb"] else None
        for s in ("t1", "t2"):
            d = df[df["split"] == s].reset_index(drop=True)
            dl = loader(d, cfg["res"], False, micro * 2, sog=cfg["sog"], mean=mean, std=std, workers=8)
            p = prever(model, dl, emb[d["linha"].to_numpy()] if emb is not None else None, tta=cfg["tta"])
            np.savez_compressed(PREV / f"{rid}_{s}.npz", isic_id=d["isic_id"].to_numpy(), p=p)
            print(f"{rid} {s}: {len(d)} imagens", flush=True)
        del model
        torch.cuda.empty_cache()


def carregar(rid):
    v = np.load(PREV / f"{rid}.npz")
    t1 = np.load(PREV / f"{rid}_t1.npz", allow_pickle=True)
    t2 = np.load(PREV / f"{rid}_t2.npz", allow_pickle=True)
    return v, t1, t2


def metricas_teste(rid, df):
    v, t1, t2 = carregar(rid)
    tau, limiar, lp = float(v["tau"]), float(v["limiar"]), v["log_prior"]
    d1 = df.set_index("isic_id").loc[t1["isic_id"]]
    d2 = df.set_index("isic_id").loc[t2["isic_id"]]
    y1 = d1["dx_ham"].map({c: i for i, c in enumerate(HAM_CLASSES)}).to_numpy()
    y2 = d2["dx"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    r = {"id": rid}
    r |= avaliar(v["p"], v["y"], tau, lp, limiar=limiar, prefixo="val_")
    r |= avaliar(t1["p"], y1, tau, lp, limiar=limiar, ham=True, prefixo="t1_")
    r |= avaliar(t2["p"], y2, tau, lp, limiar=limiar, prefixo="t2_")
    return r, (t1["p"], y1, tau, lp), (t2["p"], y2, tau, lp)


def bootstrap(a, b, ham, n=2000, seed=SEED):
    """Diferença (b − a) com IC 95%, reamostrando as MESMAS imagens para os dois modelos."""
    pa, y, ta, lpa = a
    pb, _, tb, lpb = b
    rng = np.random.default_rng(seed)
    nomes = ["bal_acc", "spec95", "mel_auc", "mal_auc"]
    difs = {k: [] for k in nomes}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < len(np.unique(y)):
            continue
        ra = avaliar(pa[idx], y[idx], ta, lpa, ham=ham)
        rb = avaliar(pb[idx], y[idx], tb, lpb, ham=ham)
        for k in nomes:
            difs[k].append(rb[k] - ra[k])
    return {k: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in difs.items()}


def dermatologistas(rids, df):
    """Leituras sem ajuda da IA (Tschandl et al., 2020) x modelos, nas mesmas imagens do T1."""
    leit = pd.read_csv(RAW / "ISIC2018_Task3_Test_NatureMedicine_AI_Interaction_Benefit.csv", decimal=",")
    cols = [f"user_dx_without_interaction_{c}" for c in HAM_CLASSES]
    leit["dx_medico"] = leit[cols].to_numpy().argmax(1)
    verdade = df.set_index("isic_id")["dx_ham"].map({c: i for i, c in enumerate(HAM_CLASSES)})
    leit["y"] = leit["image_id"].map(verdade)
    leit = leit.dropna(subset=["y"])
    y = leit["y"].astype(int).to_numpy()
    mal = np.isin(y, [HAM_CLASSES.index(c) for c in ["akiec", "bcc", "mel"]])
    pm = np.isin(leit["dx_medico"], [HAM_CLASSES.index(c) for c in ["akiec", "bcc", "mel"]])
    from sklearn.metrics import balanced_accuracy_score
    linhas = [{"quem": "dermatologistas (sem IA)", "leituras": len(leit), "acc": float((leit["dx_medico"] == y).mean()),
               "bal_acc": balanced_accuracy_score(y, leit["dx_medico"]),
               "sens_maligno": float(pm[mal].mean()), "spec_maligno": float((~pm[~mal]).mean())}]
    for rid in rids:
        v, t1, _ = carregar(rid)
        idx = pd.Series(np.arange(len(t1["isic_id"])), index=t1["isic_id"])[leit["image_id"]].to_numpy()
        p = t1["p"][idx]
        pred = para_ham(ajustar(p, float(v["tau"]), v["log_prior"])).argmax(1)
        s = escore_maligno(p)
        pr = s >= float(v["limiar"])
        linhas.append({"quem": rid, "leituras": len(leit), "acc": float((pred == y).mean()),
                       "bal_acc": balanced_accuracy_score(y, pred),
                       "sens_maligno": float(pr[mal].mean()), "spec_maligno": float((~pr[~mal]).mean())})
    return pd.DataFrame(linhas)


def relatorio(base, finalistas):
    df = ler_splits()
    linhas, boots = [], []
    r0, t1_0, t2_0 = metricas_teste(base, df)
    linhas.append(r0)
    for rid in finalistas:
        r, t1, t2 = metricas_teste(rid, df)
        linhas.append(r)
        for nome, a, b, ham in (("t1", t1_0, t1, True), ("t2", t2_0, t2, False)):
            for k, (m, lo, hi) in bootstrap(a, b, ham).items():
                boots.append({"modelo": rid, "teste": nome, "metrica": k, "dif_vs_base": m, "ic_inf": lo, "ic_sup": hi,
                              "significativo": lo > 0 or hi < 0})
    out = EXPERIMENTOS.parent
    pd.DataFrame(linhas).to_csv(out / "testes.csv", index=False)
    pd.DataFrame(boots).to_csv(out / "bootstrap.csv", index=False)
    dermatologistas([base] + finalistas, df).to_csv(out / "dermatologistas.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["id", "val_bal_acc", "t1_bal_acc", "t1_spec95", "t1_mel_auc", "t2_bal_acc", "t2_spec95", "t2_mel_auc"]
    print(pd.DataFrame(linhas)[cols].round(4).to_string(index=False))
    print(pd.DataFrame(boots).round(4).to_string(index=False))
    print(pd.read_csv(out / "dermatologistas.csv").round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prever")
    a.add_argument("ids", nargs="+")
    b = sub.add_parser("relatorio")
    b.add_argument("--base", required=True)
    b.add_argument("--finalistas", nargs="+", required=True)
    args = ap.parse_args()
    if args.cmd == "prever":
        prever_testes(args.ids)
    else:
        relatorio(args.base, args.finalistas)

"""E5: ensemble dos melhores modelos de E1 a E4, com calibração por temperatura (CPU, minutos).

Uso:
    python -m dermato.ensemble E1_final_s42 E2_final_s42 E3_final_s42 E4_final_s42 [--testes]

Como usa as previsões dos outros modelos na própria validação, o ensemble é AJUSTADO na metade A
da validação (pesos, LightGBM, temperatura) e AVALIADO na metade B (divisão fixa por grupo,
semente 42). Testa todos os subconjuntos de modelos e três métodos: média, média ponderada e
LightGBM. Com --testes, aplica o vencedor às previsões de teste (T1/T2) já geradas por testar.py.
"""
import argparse
import itertools
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupShuffleSplit

from .config import CLASSES, MAL_IDX, RESULTADOS, SEED
from .dados import ler_splits
from .metricas import avaliar, escolher_tau, escore_maligno, limiar_sens, softmax
from .treinar import registrar

PREV = RESULTADOS / "previsoes"


def metades(va):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    a, b = next(gss.split(va, groups=va["grupo"]))
    return a, b


def temperatura(logp, y):
    def nll(t):
        p = softmax(logp / t[0])
        return -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    return float(minimize(nll, [1.0], bounds=[(0.05, 10)]).x[0])


def pesos_otimos(ps, y, log_prior):
    """Pesos na simplex que minimizam a log-verossimilhança na metade A."""
    k = len(ps)

    def nll(w):
        w = np.abs(w) / np.abs(w).sum()
        p = sum(wi * pi for wi, pi in zip(w, ps))
        return -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    w = minimize(nll, np.full(k, 1 / k), method="Nelder-Mead").x
    return np.abs(w) / np.abs(w).sum()


def combinar(ps_a, ps_b, y_a, metodo, log_prior):
    if metodo == "media":
        return sum(ps_a) / len(ps_a), sum(ps_b) / len(ps_b), {}
    if metodo == "ponderada":
        w = pesos_otimos(ps_a, y_a, log_prior)
        return sum(wi * p for wi, p in zip(w, ps_a)), sum(wi * p for wi, p in zip(w, ps_b)), {"pesos": w.tolist()}
    import lightgbm as lgb
    Xa, Xb = np.log(np.concatenate(ps_a, 1) + 1e-6), np.log(np.concatenate(ps_b, 1) + 1e-6)
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=10,
                           class_weight="balanced", random_state=SEED, verbose=-1).fit(Xa, y_a)
    return m.predict_proba(Xa), m.predict_proba(Xb), {"modelo": m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--testes", action="store_true")
    args = ap.parse_args()

    df = ler_splits()
    va = df[df["split"] == "val"].reset_index(drop=True)
    prev = {}
    for rid in args.ids:
        z = np.load(PREV / f"{rid}.npz", allow_pickle=True)
        ordem = pd.Series(np.arange(len(z["isic_id"])), index=z["isic_id"])[va["isic_id"]].to_numpy()
        prev[rid] = (z["p"][ordem], z["log_prior"])
    y = va["y"].to_numpy()
    ia, ib = metades(va)
    lp_uniforme = np.log(np.full(len(CLASSES), 1 / len(CLASSES)))

    linhas = []
    for k in range(1, len(args.ids) + 1):
        for sub in itertools.combinations(args.ids, k):
            for metodo in (["media"] if k == 1 else ["media", "ponderada", "lightgbm"]):
                # cada membro com o prior do próprio treino retirado (tau escolhido na metade A)
                ps_a, ps_b = [], []
                for rid in sub:
                    p, lp = prev[rid]
                    t = escolher_tau(p[ia], y[ia], lp)
                    adj = softmax(np.log(p + 1e-12) - t * lp)
                    ps_a.append(adj[ia])
                    ps_b.append(adj[ib])
                pa, pb, extra = combinar(ps_a, ps_b, y[ia], metodo, lp_uniforme)
                T = temperatura(np.log(pa + 1e-12), y[ia])
                pa, pb = softmax(np.log(pa + 1e-12) / T), softmax(np.log(pb + 1e-12) / T)
                tau = escolher_tau(pa, y[ia], lp_uniforme)
                lim = limiar_sens(np.isin(y[ia], MAL_IDX), escore_maligno(pa))
                r = avaliar(pb, y[ib], tau, lp_uniforme, limiar=lim)
                linhas.append({"membros": "+".join(sub), "metodo": metodo, "temperatura": T,
                               "extra": json.dumps({k2: v for k2, v in extra.items() if k2 != "modelo"})} | r)
    tab = pd.DataFrame(linhas).sort_values("bal_acc", ascending=False)
    tab.to_csv(RESULTADOS / "ensemble_busca.csv", index=False)
    pd.set_option("display.width", 250)
    print(tab[["membros", "metodo", "bal_acc", "spec95", "mel_auc", "mal_auc", "ece", "temperatura"]]
          .head(15).round(4).to_string(index=False))

    melhor = tab.iloc[0]
    print(f"\nVencedor (metade B da validação): {melhor['membros']} / {melhor['metodo']}")
    registrar({"id": "E5_ensemble", "etapa": "E5", "familia": f"ensemble_{melhor['metodo']}", "seed": SEED,
               "status": "ok", "config": json.dumps({"membros": melhor["membros"], "metodo": melhor["metodo"],
                                                     "temperatura": melhor["temperatura"], "extra": melhor["extra"]})}
              | {f"val_{k}": melhor[k] for k in ["bal_acc", "spec95", "mel_auc", "mal_auc", "ece", "tau",
                                                 "triagem_sens", "triagem_spec"]})
    # previsões do ensemble na validação inteira (para testar.py ler tau/limiar do mesmo jeito que os outros)
    salvar_ensemble(melhor, prev, va, y, ia, lp_uniforme, args.testes)


def salvar_ensemble(melhor, prev, va, y, ia, lp_uniforme, testes):
    sub = melhor["membros"].split("+")
    metodo = melhor["metodo"]
    taus = {rid: escolher_tau(prev[rid][0][ia], y[ia], prev[rid][1]) for rid in sub}

    def ajust(rid, p):
        return softmax(np.log(p + 1e-12) - taus[rid] * prev[rid][1])

    ps = [ajust(rid, prev[rid][0]) for rid in sub]
    conjuntos = {"": ps}
    if testes:
        for s in ("t1", "t2"):
            conjuntos["_" + s] = [ajust(rid, np.load(PREV / f"{rid}_{s}.npz")["p"]) for rid in sub]
    ps_a = [p[ia] for p in ps]
    for suf, lista in conjuntos.items():
        _, p, _ = combinar(ps_a, lista, y[ia], metodo, lp_uniforme)
        p = softmax(np.log(p + 1e-12) / melhor["temperatura"])
        if suf == "":
            tau = escolher_tau(p[ia], y[ia], lp_uniforme)
            lim = limiar_sens(np.isin(y[ia], MAL_IDX), escore_maligno(p[ia]))
            np.savez_compressed(PREV / "E5_ensemble.npz", isic_id=va["isic_id"].to_numpy(), p=p, y=y,
                                log_prior=lp_uniforme, tau=tau, limiar=lim)
        else:
            ids = np.load(PREV / f"{sub[0]}{suf}.npz", allow_pickle=True)["isic_id"]
            np.savez_compressed(PREV / f"E5_ensemble{suf}.npz", isic_id=ids, p=p)
    print("Previsões do ensemble salvas em", PREV)


if __name__ == "__main__":
    main()

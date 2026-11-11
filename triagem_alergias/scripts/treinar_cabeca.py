# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "pandas", "numpy", "scikit-learn"]
# ///
"""Cabeça de classificação sobre embeddings congelados (iteração 1+).

Uso:  uv run triagem_alergias/scripts/treinar_cabeca.py --embeddings dinov2_base [dermlip_panderm ...]
          [--meta] [--rotulo unico|suave] [--arq linear|mlp] [--nome NOME]

Treina no split train, escolhe hiperparâmetros (weight decay, épocas) na validação pelo F1 macro
(mesmo critério do notebook) e mede no teste uma vez. Guarda probs e métricas em
triagem_alergias/experimentos/<nome>/.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from dados_scin import PASTA, tabela_especialista  # noqa: E402
from metricas import metricas_especialista, resumo  # noqa: E402

EMB = PASTA / "experimentos" / "embeddings"
SEED = 42


def pesos_classes(y, n):
    counts = np.bincount(y, minlength=n).astype(float)
    w = np.sqrt(counts.sum() / np.maximum(counts, 1))
    return torch.tensor(w / w[y].mean(), dtype=torch.float32)


def treinar(Xtr, Ttr, ytr, Xva, yva, n_classes, arq, wd, epocas, lr=1e-3, oculto=512, dropout=0.3, seed=SEED):
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    if arq == "linear":
        model = torch.nn.Sequential(torch.nn.Dropout(dropout), torch.nn.Linear(d, n_classes))
    else:
        model = torch.nn.Sequential(torch.nn.Dropout(dropout), torch.nn.Linear(d, oculto), torch.nn.GELU(),
                                    torch.nn.Dropout(dropout), torch.nn.Linear(oculto, n_classes))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    w = pesos_classes(ytr, n_classes)
    Xtr_t, Ttr_t, Xva_t = torch.tensor(Xtr), torch.tensor(Ttr), torch.tensor(Xva)
    ytr_t = torch.tensor(ytr)
    n = len(Xtr)
    melhor, melhor_estado, hist = -1, None, []
    from sklearn.metrics import f1_score
    for ep in range(epocas):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            logits = model(Xtr_t[idx])
            perda = -(Ttr_t[idx] * F.log_softmax(logits, 1)).sum(1)  # entropia cruzada com alvo (suave ou one-hot)
            perda = (perda * w[ytr_t[idx]]).mean()
            opt.zero_grad()
            perda.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva_t).argmax(1).numpy()
        f1 = f1_score(yva, pv, average="macro", zero_division=0)
        hist.append(f1)
        if f1 > melhor:
            melhor, melhor_estado = f1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(melhor_estado)
    model.eval()
    return model, melhor, int(np.argmax(hist)) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", nargs="*", default=[], help="vazio = só metadados")
    ap.add_argument("--meta", action="store_true")
    ap.add_argument("--caso", action="store_true", help="concatena a média dos embeddings das fotos do mesmo caso")
    ap.add_argument("--rotulo", choices=["unico", "suave"], default="suave")
    ap.add_argument("--arq", choices=["linear", "mlp"], default="linear")
    ap.add_argument("--epocas", type=int, default=60)
    ap.add_argument("--nome")
    args = ap.parse_args()
    torch.set_num_threads(8)

    df, classes, alvos, meta = tabela_especialista()
    n = len(classes)
    y = df["classe"].map({c: i for i, c in enumerate(classes)}).values
    partes = []
    for e in args.embeddings:
        X = np.load(EMB / f"{e}.npy").astype("float32")
        ids = pd.read_csv(EMB / f"{e}_ids.csv", dtype=str)
        assert (ids["image_id"].values == df["image_id"].values).all(), f"ordem de {e} não bate"
        X = X / np.linalg.norm(X, axis=1, keepdims=True)  # normaliza cada modelo antes de concatenar
        partes.append(X)
        if args.caso:
            media = pd.DataFrame(X).groupby(df["case_id"].values).transform("mean").values.astype("float32")
            partes.append(media)
    if args.meta:
        partes.append(meta.values.astype("float32"))
    X = np.concatenate(partes, 1)
    tr, va, te = [(df["split"] == s).values for s in ("train", "val", "test")]
    # padronização ajustada só no treino
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    X = (X - mu) / sd
    T = alvos if args.rotulo == "suave" else np.eye(n, dtype="float32")[y]

    assert partes, "passe --embeddings e/ou --meta"
    nome = args.nome or ("+".join(args.embeddings + (["caso"] if args.caso else []) + (["meta"] if args.meta else []))
                         + f"_{args.rotulo}_{args.arq}")
    out = PASTA / "experimentos" / nome
    out.mkdir(parents=True, exist_ok=True)
    busca = []
    for wd in (1e-4, 1e-3, 1e-2, 1e-1):
        for dropout in (0.3, 0.6):
            model, f1v, ep = treinar(X[tr], T[tr], y[tr], X[va], y[va], n, args.arq, wd, args.epocas, dropout=dropout)
            busca.append({"wd": wd, "dropout": dropout, "val_f1_macro": f1v, "melhor_epoca": ep})
            print(f"  wd {wd:g} dropout {dropout}: val F1 macro {f1v:.4f} (época {ep})", flush=True)
    melhor = max(busca, key=lambda b: b["val_f1_macro"])
    # retreina com os melhores hiperparâmetros (a seleção de época continua pela validação)
    model, f1v, ep = treinar(X[tr], T[tr], y[tr], X[va], y[va], n, args.arq, melhor["wd"], args.epocas, dropout=melhor["dropout"])
    res = {"config": vars(args) | {"escolhido": melhor}, "busca": busca}
    with torch.no_grad():
        for split, m in (("val", va), ("test", te)):
            probs = F.softmax(model(torch.tensor(X[m])), 1).numpy()
            np.save(out / f"probs_{split}.npy", probs)
            res[split] = metricas_especialista(probs, y[m], df["case_id"][m].values, classes)
            print(split, resumo(res[split]), flush=True)
    torch.save({"state": model.state_dict(), "mu": mu, "sd": sd, "classes": classes, "config": vars(args)}, out / "cabeca.pt")
    (out / "metricas.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("salvo em", out)


if __name__ == "__main__":
    main()

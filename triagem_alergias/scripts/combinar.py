"""Combina as probabilidades de experimentos já rodados (média ponderada, pesos escolhidos na validação).

Uso:  py -3.12 triagem_alergias/scripts/combinar.py exp1 exp2 [exp3 ...] [--nome NOME]
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from dados_scin import PASTA, tabela_especialista  # noqa: E402
from metricas import metricas_especialista, resumo  # noqa: E402

EXP = PASTA / "experimentos"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experimentos", nargs="+")
    ap.add_argument("--nome")
    args = ap.parse_args()
    df, classes, _, _ = tabela_especialista()
    y = df["classe"].map({c: i for i, c in enumerate(classes)}).values
    va, te = (df["split"] == "val").values, (df["split"] == "test").values
    P = {s: [np.load(EXP / e / f"probs_{s}.npy") for e in args.experimentos] for s in ("val", "test")}
    grade = [w for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=len(args.experimentos)) if abs(sum(w) - 1) < 1e-6]
    melhor, melhor_w = -1, None
    for w in grade:
        p = sum(wi * pi for wi, pi in zip(w, P["val"]))
        f1 = metricas_especialista(p, y[va], df["case_id"][va].values, classes)["f1_macro"]
        if f1 > melhor:
            melhor, melhor_w = f1, w
    nome = args.nome or "combinacao_" + "+".join(args.experimentos)
    out = EXP / nome
    out.mkdir(exist_ok=True)
    res = {"pesos": dict(zip(args.experimentos, [float(x) for x in melhor_w]))}
    print("pesos (validação, F1 macro):", res["pesos"])
    for s, m in (("val", va), ("test", te)):
        p = sum(wi * pi for wi, pi in zip(melhor_w, P[s]))
        np.save(out / f"probs_{s}.npy", p)
        res[s] = metricas_especialista(p, y[m], df["case_id"][m].values, classes)
        print(s, resumo(res[s]))
    (out / "metricas.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

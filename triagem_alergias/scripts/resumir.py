"""Tabela com todos os experimentos do especialista (triagem_alergias/experimentos/*/metricas.json)."""
import json
import sys
from pathlib import Path

import pandas as pd

PASTA = Path(__file__).resolve().parents[1] / "experimentos"
linhas = []
for f in sorted(PASTA.glob("*/metricas.json")):
    m = json.loads(f.read_text(encoding="utf-8"))
    for split in ("val", "test"):
        r = m.get(split)
        if r:
            linhas.append({"experimento": f.parent.name, "split": split, "top1": r["acuracia"], "top3": r["acuracia_top3"],
                           "bal_acc": r["acuracia_balanceada"], "f1_macro": r["f1_macro"], "auroc": r["auroc_macro"],
                           "por_caso": r["acuracia_por_caso"]})
t = pd.DataFrame(linhas)
split = sys.argv[1] if len(sys.argv) > 1 else "test"
t = t[t["split"] == split].drop(columns="split").sort_values("f1_macro", ascending=False)
pd.set_option("display.width", 200)
print(f"== {split} ==")
print(t.round(3).to_string(index=False))
(PASTA / f"resumo_{split}.md").write_text(t.round(3).to_markdown(index=False), encoding="utf-8")

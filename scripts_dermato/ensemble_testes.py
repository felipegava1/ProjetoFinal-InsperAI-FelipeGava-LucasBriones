"""Aplica os ensembles já escolhidos na validação (E5_ensemble, E5_OT) às previsões de teste T1/T2.

Lê membros, método e temperatura da linha registrada em experimentos.csv (nada é reescolhido) e
reaproveita `ensemble.salvar_ensemble`, que grava <nome>.npz, <nome>_t1.npz e <nome>_t2.npz.
Uso: python scripts_dermato/ensemble_testes.py E5_ensemble E5_OT
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dermato.config import CLASSES, EXPERIMENTOS  # noqa: E402
from dermato.dados import ler_splits  # noqa: E402
from dermato.ensemble import PREV, metades, salvar_ensemble  # noqa: E402

df = ler_splits()
va = df[df["split"] == "val"].reset_index(drop=True)
y = va["y"].to_numpy()
ia, _ = metades(va)
lp_uniforme = np.log(np.full(len(CLASSES), 1 / len(CLASSES)))
exp = pd.read_csv(EXPERIMENTOS)

for nome in sys.argv[1:]:
    melhor = json.loads(exp[exp["id"] == nome].iloc[-1]["config"])
    prev = {}
    for rid in melhor["membros"].split("+"):
        z = np.load(PREV / f"{rid}.npz", allow_pickle=True)
        ordem = pd.Series(np.arange(len(z["isic_id"])), index=z["isic_id"])[va["isic_id"]].to_numpy()
        prev[rid] = (z["p"][ordem], z["log_prior"])
    salvar_ensemble(melhor, prev, va, y, ia, lp_uniforme, True, nome)
    print(nome, "->", melhor["membros"], melhor["metodo"])

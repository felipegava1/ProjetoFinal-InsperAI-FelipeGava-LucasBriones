"""Otimização do melhor modelo da v2 (roda sozinho depois do retreino; 14/11).

1. Espera o retreino v2 terminar (linha "Retreino v2 concluído" em F:/dermato/retreino_v2.log).
2. Melhor modelo = etapa com maior acurácia balanceada média (entre sementes) na validação v2.
3. Busca OT (Optuna) em volta da config dele até ~07:30, 1ª tentativa = a própria config.
   Retreino da vencedora (OT_v2_final_s42) + sementes 43 e 44.
4. Ensemble E5_OT: os membros da E5 com o modelo da família vencedora trocado pelo otimizado.
5. Análise de erros (F1/F2 por classe, erros tipo I/II) de base, otimizado e ensembles.

    python scripts_dermato/otimizar_v2.py [--horas H]
"""
import argparse
import datetime as dt
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from dermato.config import EXPERIMENTOS, RESULTADOS  # noqa: E402

import pandas as pd  # noqa: E402

LOG_RETREINO = Path(r"F:\dermato\retreino_v2.log")
LOGS = RESULTADOS / "logs"
MEMBROS_E5 = ["E1_final_s42", "E2_384_final_s42", "E3_final_s42", "E4_final_s42"]


def agora():
    return time.strftime("%d/%m %H:%M")


def rodar(nome, args):
    LOGS.mkdir(parents=True, exist_ok=True)
    print(f"[{agora()}] >>> {nome}: {' '.join(args)[:200]}", flush=True)
    t0 = time.time()
    with open(LOGS / f"ot_{nome}.log", "a", encoding="utf-8") as log:
        r = subprocess.run([sys.executable, "-u", *args], stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
    print(f"[{agora()}] <<< {nome}: código {r.returncode}, {(time.time() - t0) / 60:.0f} min", flush=True)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horas", type=float, help="duração da busca (padrão: até as 07:30)")
    ap.add_argument("--nao-esperar", action="store_true")
    ap.add_argument("--vencedor", help="força a etapa a otimizar (ex.: E1)")
    a = ap.parse_args()

    if not a.nao_esperar:
        print(f"[{agora()}] esperando o retreino v2...", flush=True)
        while "Retreino v2 concluído" not in (LOG_RETREINO.read_text(encoding="utf-8", errors="ignore")
                                              if LOG_RETREINO.exists() else ""):
            time.sleep(120)

    # 2. melhor modelo
    exp = pd.read_csv(EXPERIMENTOS)
    fin = exp[exp["id"].str.match(r"^E\d\w*_final_s\d+$", na=False) & (exp["status"] == "ok")].copy()
    fin["etapa_base"] = fin["id"].str.replace(r"_final_s\d+$", "", regex=True)
    tab = fin.groupby("etapa_base").agg(n=("val_bal_acc", "size"), bal_acc=("val_bal_acc", "mean"),
                                        spec95=("val_spec95", "mean"), mal_auc=("val_mal_auc", "mean"))
    tab = tab.sort_values("bal_acc", ascending=False)
    print(tab.round(4).to_string(), flush=True)
    # média de 3 sementes vale mais que uma rodada só (ruído medido ~1 ponto): uma etapa com uma semente
    # só vence se passar a melhor média de 3 sementes por mais de 1 ponto
    tres = tab[tab["n"] >= 3]
    vencedor = tres.index[0]
    um = tab[(tab["n"] < 3) & (tab["bal_acc"] > tres["bal_acc"].iloc[0] + 0.01)]
    if len(um):
        vencedor = um.index[0]
    if a.vencedor:
        vencedor = a.vencedor
    base = f"{vencedor}_final_s42"
    print(f"[{agora()}] melhor modelo: {vencedor} (base {base})", flush=True)

    # 3. busca
    if a.horas:
        horas = a.horas
    else:
        fim = dt.datetime.now().replace(hour=7, minute=30, second=0, microsecond=0)
        if fim < dt.datetime.now():
            fim += dt.timedelta(days=1)
        horas = max(3.0, (fim - dt.datetime.now()).total_seconds() / 3600)
    extra = []
    cfg_base = json.loads(exp.set_index("id").loc[base, "config"])
    if cfg_base.get("pesos_cnn"):
        extra = ["--pesos-cnn", cfg_base["pesos_cnn"]]
    if cfg_base.get("emb"):
        extra += ["--emb", cfg_base["emb"]]
    if not (RESULTADOS / "modelos" / "OT_v2_final_s42.pt").exists():
        rodar("busca", ["-m", "dermato.busca", "OT", "--base", base, "--estudo", "OT_v2",
                        "--tentativas", "60", "--horas", f"{horas:.2f}", *extra])

    exp = pd.read_csv(EXPERIMENTOS).set_index("id")
    if "OT_v2_final_s42" not in exp.index:
        sys.exit("busca não gerou OT_v2_final_s42")
    cfg = json.loads(exp.loc["OT_v2_final_s42", "config"])
    for seed in (43, 44):
        rid = f"OT_v2_final_s{seed}"
        if (RESULTADOS / "modelos" / f"{rid}.pt").exists():
            continue
        c = cfg | {"seed": seed, "salvar_modelo": True, "paciencia": 99}
        rodar(rid, ["-c", f"from dermato.treinar import treinar; import json; "
                          f"treinar(json.loads({json.dumps(json.dumps(c))}), nome={rid!r})"])

    # 4. ensemble com o otimizado no lugar da família vencedora
    membros = [m for m in MEMBROS_E5 if not m.startswith(vencedor + "_final")] + ["OT_v2_final_s42"]
    if not (RESULTADOS / "previsoes" / "E5_OT.npz").exists():
        rodar("E5_OT", ["-m", "dermato.ensemble", *membros, "--nome", "E5_OT"])

    # 5. análise de erros
    ids = [base, f"{vencedor}_final_s43", f"{vencedor}_final_s44", "OT_v2_final_s42", "OT_v2_final_s43",
           "OT_v2_final_s44", "E0_final_s42", "E5_ensemble", "E5_OT"]
    ids = [i for i in ids if (RESULTADOS / "previsoes" / f"{i}.npz").exists()]
    rodar("erros", ["-m", "dermato.erros", *ids])
    print(f"[{agora()}] Otimização v2 concluída.", flush=True)


if __name__ == "__main__":
    main()

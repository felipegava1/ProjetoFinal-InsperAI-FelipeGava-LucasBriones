"""Retreino de todos os modelos finais na divisão v2 (sem imagens repetidas entre splits; 14/11).

As configurações são exatamente as vencedoras da v1 (lidas de resultados/v1_com_vazamento/experimentos.csv):
refazer as buscas do Optuna (30 tentativas por etapa) levaria vários dias. Limitação registrada no diário:
os parâmetros foram escolhidos numa validação com ~3% de imagens repetidas do treino.

Ordem: primeiro um modelo por etapa (E0, E1, E2_384, E3, E4) e o ensemble E5, depois as sementes extras
dos finalistas e as variantes. Retomável: pula as rodadas cujo modelo já existe.

    python scripts_dermato/retreinar_v2.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from dermato.config import RESULTADOS  # noqa: E402

import pandas as pd  # noqa: E402

V1 = REPO / "resultados" / "v1_com_vazamento" / "experimentos.csv"
MODELOS = RESULTADOS / "modelos"
LOGS = RESULTADOS / "logs"
E1 = str(MODELOS / "E1_final_s42.pt")
EMB = str(RESULTADOS / "panderm_emb.npz")

RODADAS = [
    ("E0_final_s42", {}),
    ("E1_final_s42", {}),
    ("E2_384_final_s42", {}),
    ("E3_final_s42", {"pesos_cnn": E1}),
    ("E4_final_s42", {"pesos_cnn": E1, "emb": EMB}),
    ("E5", None),
    ("E0_busca_final_s42", {}),
    ("E1_final_s43", {}), ("E1_final_s44", {}),
    ("E2_384_final_s43", {}), ("E2_384_final_s44", {}),
    ("E3_final_s43", {"pesos_cnn": E1}), ("E3_final_s44", {"pesos_cnn": E1}),
    ("E1r_final_s42", {}),
    ("E2_final_s42", {}),
]
MEMBROS_E5 = ["E1_final_s42", "E2_384_final_s42", "E3_final_s42", "E4_final_s42"]


def rodar(nome, args):
    LOGS.mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%d/%m %H:%M')}] >>> {nome}", flush=True)
    t0 = time.time()
    with open(LOGS / f"v2_{nome}.log", "a", encoding="utf-8") as log:
        r = subprocess.run([sys.executable, "-u", *args], stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
    print(f"[{time.strftime('%d/%m %H:%M')}] <<< {nome}: código {r.returncode}, {(time.time() - t0) / 60:.0f} min",
          flush=True)
    return r.returncode == 0


def main():
    v1 = pd.read_csv(V1).set_index("id")
    for rid, extra in RODADAS:
        if extra is None:                                       # ensemble
            if (RESULTADOS / "previsoes" / "E5_ensemble.npz").exists():
                continue
            if all((MODELOS / f"{m}.pt").exists() for m in MEMBROS_E5):
                rodar("E5", ["-m", "dermato.ensemble", *MEMBROS_E5])
            continue
        if (MODELOS / f"{rid}.pt").exists():
            continue
        base = rid.rsplit("_s", 1)[0] + "_s42"                  # sementes extras usam a config da s42
        cfg = json.loads(v1.loc[base, "config"])
        cfg |= {"seed": int(rid.rsplit("_s", 1)[1]), "salvar_modelo": True, "paciencia": 99} | extra
        if extra.get("pesos_cnn") and not Path(extra["pesos_cnn"]).exists():
            print(f"pulando {rid}: falta {extra['pesos_cnn']}", flush=True)
            continue
        codigo = (f"from dermato.treinar import treinar; import json; "
                  f"treinar(json.loads({json.dumps(json.dumps(cfg))}), nome={rid!r})")
        rodar(rid, ["-c", codigo])
    print("Retreino v2 concluído.", flush=True)


if __name__ == "__main__":
    main()

"""Fila de execução do plano (uma etapa por vez na GPU). Retomável: pula o que já terminou.

Uso:
    python -m dermato.fila                   # tudo, de onde parou
    python -m dermato.fila --ate E1          # para depois da etapa indicada
    python -m dermato.fila --tentativas 15   # corte de tempo previsto no plano (30 -> 15)

Ordem: cache -> E0 -> vetores PanDerm + teste rápido -> E1 -> E2 -> E3 -> E4 -> E5 -> finalistas
(3 melhores com mais 2 sementes). Os testes T1/T2 NÃO são abertos aqui: isso é o dia 6
(python -m dermato.testar ...), depois que os finalistas estiverem fechados.
Logs em RESULTADOS/logs/<passo>.log.
"""
import argparse
import json
import subprocess
import sys
import time

import pandas as pd

from .config import EXPERIMENTOS, RESULTADOS, SEED

PY = sys.executable
LOGS = RESULTADOS / "logs"
MODELOS = RESULTADOS / "modelos"
ORDEM = ["cache", "E0", "E0_busca", "panderm", "E1", "E2", "E1r", "E2_384", "E3", "E4", "E5", "finalistas"]


def rodar(nome, args):
    LOGS.mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%d/%m %H:%M')}] >>> {nome}: {' '.join(args)}", flush=True)
    t0 = time.time()
    with open(LOGS / f"{nome}.log", "a", encoding="utf-8") as log:
        r = subprocess.run([PY, "-u", *args], stdout=log, stderr=subprocess.STDOUT)
    print(f"[{time.strftime('%d/%m %H:%M')}] <<< {nome}: código {r.returncode}, {(time.time() - t0) / 60:.0f} min",
          flush=True)
    if r.returncode != 0:
        sys.exit(f"Falhou em {nome}; veja {LOGS / (nome + '.log')}")


def final(etapa):
    return MODELOS / f"{etapa}_final_s{SEED}.pt"


def melhor(*etapas):
    """Entre variantes de uma etapa (ex.: E1 e E1r), a de maior acurácia balanceada na validação."""
    prontas = [e for e in etapas if final(e).exists()]
    if not prontas:
        return None
    exp = pd.read_csv(EXPERIMENTOS).set_index("id")
    return max(prontas, key=lambda e: exp.loc[f"{e}_final_s{SEED}", "val_bal_acc"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ate", choices=ORDEM, default="finalistas")
    ap.add_argument("--tentativas", type=int, default=30)
    ap.add_argument("--sem-e2", action="store_true", help="corte previsto no plano se faltar tempo")
    ap.add_argument("--sem-panderm", action="store_true")
    ap.add_argument("--horas", default="", help='limite por busca, ex.: "E0_busca=1.5,E1=3.5,E2=3.5"')
    a = ap.parse_args()
    passos = ORDEM[:ORDEM.index(a.ate) + 1]
    t = str(a.tentativas)
    horas = dict(x.split("=") for x in a.horas.split(",") if x)

    def lim(etapa):
        return ["--horas", horas[etapa]] if etapa in horas else []

    if "cache" in passos:
        rodar("cache", ["-m", "dermato.preparar", "cache"])
    if "E0" in passos and not final("E0").exists():
        # receita atual: 8 épocas, paciência 3, escolha da época pela validação
        rodar("E0", ["-m", "dermato.treinar", "--etapa", "E0", "--salvar", "--nome", f"E0_final_s{SEED}"])
    if "E0_busca" in passos and not final("E0_busca").exists():
        rodar("E0_busca", ["-m", "dermato.busca", "E0_busca", "--tentativas", t, *lim("E0_busca")])
    if "panderm" in passos and not a.sem_panderm:
        if not (RESULTADOS / "panderm_emb.npz").exists():
            rodar("panderm_extrair", ["-m", "dermato.panderm", "extrair"])
        if not (RESULTADOS / "previsoes" / "PanDerm_rapido_lightgbm.npz").exists():
            rodar("panderm_rapido", ["-m", "dermato.panderm", "rapido"])
    for etapa in ("E1", "E2", "E1r", "E2_384", "E3", "E4"):
        if etapa not in passos or final(etapa).exists() or (etapa == "E2" and a.sem_e2):
            continue
        extra = []
        if etapa in ("E3", "E4"):
            extra += ["--pesos-cnn", str(final(melhor("E1", "E1r")))]   # a melhor E1
        if etapa == "E4":
            extra += ["--emb", str(RESULTADOS / "panderm_emb.npz")]
        rodar(etapa, ["-m", "dermato.busca", etapa, "--tentativas", t, *lim(etapa), *extra])
    familias = [melhor("E1", "E1r"), melhor("E2", "E2_384"), melhor("E3"), melhor("E4")]
    membros = [f"{e}_final_s{SEED}" for e in familias if e]
    if "E5" in passos and not (RESULTADOS / "previsoes" / "E5_ensemble.npz").exists():
        rodar("E5", ["-m", "dermato.ensemble", *membros])
    if "finalistas" in passos:
        exp = pd.read_csv(EXPERIMENTOS)
        fin = exp[exp["id"].isin(membros)].sort_values("val_bal_acc", ascending=False).head(3)
        for _, linha in fin.iterrows():
            cfg = json.loads(linha["config"])
            for seed in (SEED + 1, SEED + 2):
                rid = f"{linha['etapa']}_final_s{seed}"
                if (MODELOS / f"{rid}.pt").exists():
                    continue
                cfg_s = cfg | {"seed": seed, "salvar_modelo": True, "paciencia": 99}
                rodar(rid, ["-c", f"from dermato.treinar import treinar; import json; "
                                  f"treinar(json.loads({json.dumps(json.dumps(cfg_s))}), nome={rid!r})"])
        print("Finalistas fechados. Próximo passo (dia 6): python -m dermato.testar prever ...")


if __name__ == "__main__":
    main()

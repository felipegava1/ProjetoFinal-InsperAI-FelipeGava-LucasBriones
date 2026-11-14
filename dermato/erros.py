"""Análise de erros: F1/F2 por classe e erros tipo I/II (falso positivo / falso negativo) na triagem.

    python -m dermato.erros E1_final_s42 OT_v2_final_s42 E5_ensemble ...

Para cada modelo (previsões salvas da validação):
  * por classe (8): precisão, sensibilidade (recall), F1 e F2 (o F2 pesa a sensibilidade 4x mais que a
    precisão: perder um câncer é pior que um alarme falso), e as médias macro;
  * maligno (ak, bcc, mel, scc) x benigno e melanoma x resto, em dois pontos de operação:
      - "triagem": escore maligno >= limiar escolhido para 95% de sensibilidade (o uso pretendido: fila
        prioritária);
      - "classe": a classe prevista (após o ajuste de prior) é maligna / é melanoma;
    com VP, FP, FN, VN, erro tipo I (taxa de falso positivo = 1 - especificidade), erro tipo II (taxa de
    falso negativo = 1 - sensibilidade), precisão, F1 e F2.
Saídas: resultados/erros_classes.csv e resultados/erros_binario.csv.
Atenção: o limiar de triagem é escolhido na própria validação (no ensemble, na metade A), então a
sensibilidade da triagem na validação fica perto de 95% por construção; o número honesto vem do teste.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_recall_fscore_support

from .config import CLASSES, IDX, MAL_IDX, REPO, achar
from .metricas import ajustar, escore_maligno


def _binario(y, pred, nome, ponto, modelo):
    vp = int((y & pred).sum()); fp = int((~y & pred).sum())
    fn = int((y & ~pred).sum()); vn = int((~y & ~pred).sum())
    prec = vp / max(vp + fp, 1); sens = vp / max(vp + fn, 1)
    f = lambda b: (1 + b * b) * prec * sens / max(b * b * prec + sens, 1e-12)
    return {"modelo": modelo, "tarefa": nome, "ponto": ponto, "VP": vp, "FP": fp, "FN": fn, "VN": vn,
            "erro_tipo_I": fp / max(fp + vn, 1), "erro_tipo_II": fn / max(fn + vp, 1),
            "sensibilidade": sens, "especificidade": vn / max(vn + fp, 1), "precisao": prec,
            "F1": f(1), "F2": f(2)}


def analisar(rid):
    z = np.load(achar(f"{rid}.npz", "previsoes"), allow_pickle=True)
    p, y = z["p"], z["y"]
    pred = ajustar(p, float(z["tau"]), z["log_prior"]).argmax(1)
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=range(len(CLASSES)), zero_division=0)
    f2 = fbeta_score(y, pred, beta=2, labels=range(len(CLASSES)), average=None, zero_division=0)
    cls = pd.DataFrame({"modelo": rid, "classe": CLASSES, "n": sup, "precisao": pr, "sensibilidade": rc,
                        "F1": f1, "F2": f2})
    cls = pd.concat([cls, pd.DataFrame([{"modelo": rid, "classe": "macro", "n": int(sup.sum()),
                                         "precisao": pr.mean(), "sensibilidade": rc.mean(),
                                         "F1": f1.mean(), "F2": f2.mean()}])], ignore_index=True)
    mal = np.isin(y, MAL_IDX)
    mel = y == IDX["mel"]
    lim = float(z["limiar"])
    binr = [
        _binario(mal, escore_maligno(p) >= lim, "maligno x benigno", "triagem (95% sens.)", rid),
        _binario(mal, np.isin(pred, MAL_IDX), "maligno x benigno", "classe prevista", rid),
        _binario(mel, pred == IDX["mel"], "melanoma x resto", "classe prevista", rid),
    ]
    return cls, pd.DataFrame(binr)


def main():
    ids = sys.argv[1:]
    cl, bi = zip(*[analisar(r) for r in ids])
    cl, bi = pd.concat(cl, ignore_index=True), pd.concat(bi, ignore_index=True)
    cl.to_csv(REPO / "resultados" / "erros_classes.csv", index=False)
    bi.to_csv(REPO / "resultados" / "erros_binario.csv", index=False)
    pd.set_option("display.width", 250)
    print(cl.pivot(index="classe", columns="modelo", values="F2").round(3).to_string())
    print(bi.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

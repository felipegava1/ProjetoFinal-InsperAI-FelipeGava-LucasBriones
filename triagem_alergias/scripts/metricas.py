"""Métricas do especialista, iguais às do triagem_alergias.ipynb (evaluate_specialist), a partir de probabilidades."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score


def metricas_especialista(probs, y, case_ids, classes):
    probs = np.asarray(probs, float)
    y = np.asarray(y)
    pred = probs.argmax(1)
    present = sorted(set(y.tolist()))
    auc = roc_auc_score(np.eye(len(classes))[y][:, present], probs[:, present], average="macro")
    by_case = pd.DataFrame(probs).groupby(np.asarray(case_ids)).mean()
    case_y = pd.Series(y).groupby(np.asarray(case_ids)).first().reindex(by_case.index).values
    case_pred = by_case.values.argmax(1)
    top3 = np.argsort(probs, axis=1)[:, -3:]
    sens = recall_score(y, pred, labels=range(len(classes)), average=None, zero_division=0)
    return {
        "n_imagens": int(len(y)), "n_casos": int(by_case.shape[0]),
        "acuracia": float(accuracy_score(y, pred)),
        "acuracia_top3": float(np.mean([y[i] in top3[i] for i in range(len(y))])),
        "acuracia_balanceada": float(balanced_accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "auroc_macro": float(auc),
        "acuracia_por_caso": float(accuracy_score(case_y, case_pred)),
        "acuracia_balanceada_por_caso": float(balanced_accuracy_score(case_y, case_pred)),
        "sensibilidade_por_classe": {c: float(s) for c, s in zip(classes, sens)},
    }


def resumo(m):
    return (f"top1 {m['acuracia']:.3f} | top3 {m['acuracia_top3']:.3f} | bal_acc {m['acuracia_balanceada']:.3f} | "
            f"f1 {m['f1_macro']:.3f} | auroc {m['auroc_macro']:.3f} | por caso {m['acuracia_por_caso']:.3f}")

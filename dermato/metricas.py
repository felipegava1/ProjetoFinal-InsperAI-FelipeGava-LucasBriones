"""Métricas do protocolo: acurácia balanceada, especificidade a 95% de sensibilidade, AUCs, ECE."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

from .config import CLASSES, HAM_CLASSES, IDX, MAL_IDX, PARA_HAM

TAUS = np.round(np.arange(0, 2.001, 0.05), 2)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ajustar(p, tau, log_prior):
    """Ajuste de prior: log p − tau·log prior (tau = 0 desliga)."""
    return softmax(np.log(p + 1e-12) - tau * log_prior)


def escolher_tau(p_val, y_val, log_prior):
    accs = [balanced_accuracy_score(y_val, ajustar(p_val, t, log_prior).argmax(1)) for t in TAUS]
    return float(TAUS[int(np.argmax(accs))])


def escore_maligno(p):
    return p[:, MAL_IDX].sum(1)


def limiar_sens(y_mal, escore, sens=0.95):
    """Maior limiar que ainda pega `sens` dos malignos."""
    return float(np.quantile(escore[y_mal], 1 - sens))


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    acertos = pred == y
    bordas = np.linspace(0, 1, bins + 1)
    total = 0.0
    for a, b in zip(bordas[:-1], bordas[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            total += m.mean() * abs(acertos[m].mean() - conf[m].mean())
    return float(total)


def para_ham(p):
    """Probabilidades de 8 classes -> 7 classes do HAM10000 (akiec = ak + scc)."""
    out = np.zeros((len(p), len(HAM_CLASSES)))
    for c, i in IDX.items():
        out[:, HAM_CLASSES.index(PARA_HAM[c])] += p[:, i]
    return out


def avaliar(p, y, tau, log_prior, limiar=None, ham=False, prefixo=""):
    """Todas as métricas de um conjunto. p: probabilidades sem ajuste (N x 8).

    y: índices em CLASSES (ou em HAM_CLASSES se ham=True). limiar: o da validação (triagem);
    se None, só a especificidade a 95% calculada no próprio conjunto.
    """
    pa = ajustar(p, tau, log_prior)
    r = {}
    if ham:
        pa_h, p_h = para_ham(pa), para_ham(p)
        pred = pa_h.argmax(1)
        r["acc"] = float((pred == y).mean())
        r["bal_acc"] = balanced_accuracy_score(y, pred)
        mel = y == HAM_CLASSES.index("mel")
        mal = np.isin(y, [HAM_CLASSES.index(c) for c in ["akiec", "bcc", "mel"]])
        r["ece"] = ece(p_h, y)
        rec = confusion_matrix(y, pred, labels=range(len(HAM_CLASSES)), normalize="true").diagonal()
        r |= {f"recall_{c}": float(v) for c, v in zip(HAM_CLASSES, rec)}
    else:
        pred = pa.argmax(1)
        r["acc"] = float((pred == y).mean())
        r["bal_acc"] = balanced_accuracy_score(y, pred)
        mel = y == IDX["mel"]
        mal = np.isin(y, MAL_IDX)
        r["ece"] = ece(p, y)
        rec = confusion_matrix(y, pred, labels=range(len(CLASSES)), normalize="true").diagonal()
        r |= {f"recall_{c}": float(v) for c, v in zip(CLASSES, rec)}
    s = escore_maligno(p)
    r["mel_auc"] = roc_auc_score(mel, p[:, IDX["mel"]])
    r["mal_auc"] = roc_auc_score(mal, s)
    lim95 = limiar_sens(mal, s)
    r["spec95"] = float((s[~mal] < lim95).mean())           # especificidade a 95% de sensibilidade
    if limiar is not None:                                   # triagem com o limiar da validação
        r["triagem_sens"] = float((s[mal] >= limiar).mean())
        r["triagem_spec"] = float((s[~mal] < limiar).mean())
    r["tau"] = tau
    return {prefixo + k: v for k, v in r.items()}

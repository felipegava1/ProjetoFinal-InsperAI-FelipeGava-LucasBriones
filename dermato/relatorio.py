"""Funções do notebook de resultados: leem só o que foi salvo (experimentos.csv, optuna.db e as
previsões de validação em resultados/previsoes/) e não retreinam nada, para o notebook rodar também
num PC sem as imagens nem GPU.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .config import CLASSES, EXPERIMENTOS, REPO, achar
from .metricas import ajustar

DB = f"sqlite:///{(REPO / 'resultados' / 'optuna.db').as_posix()}"
NOMES = {"ak": "ceratose actínica", "bcc": "carc. basocelular", "bkl": "ceratose benigna",
         "df": "dermatofibroma", "mel": "melanoma", "nv": "nevo", "scc": "carc. espinocelular",
         "vasc": "lesão vascular"}

# paleta de referência (slots categóricos validados para daltonismo; sequencial azul de um tom só)
AZUL, LARANJA = "#2a78d6", "#eb6834"
TEXTO, TEXTO2, GRADE, FUNDO = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def estilo():
    plt.rcParams.update({
        "figure.facecolor": FUNDO, "axes.facecolor": FUNDO, "savefig.facecolor": FUNDO,
        "axes.edgecolor": GRADE, "axes.labelcolor": TEXTO2, "xtick.color": TEXTO2, "ytick.color": TEXTO2,
        "text.color": TEXTO, "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
        "axes.titlesize": 11, "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "figure.dpi": 110,
    })


def experimentos():
    e = pd.read_csv(EXPERIMENTOS)
    return e


def linha_final(rid):
    e = experimentos()
    return e[e["id"] == rid].iloc[-1]


def estudo(nome):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.load_study(study_name=nome, storage=DB)


def _faixas(est):
    """Faixa testada de cada parâmetro, a partir das distribuições das tentativas."""
    dist = {}
    for t in est.trials:
        dist.update(t.distributions)
    linhas = []
    for k, d in dist.items():
        if hasattr(d, "choices"):
            faixa = ", ".join(str(c) for c in d.choices)
        else:
            log = " (log)" if getattr(d, "log", False) else ""
            faixa = f"{d.low:g} a {d.high:g}{log}"
        linhas.append({"parâmetro": k, "faixa": faixa})
    return pd.DataFrame(linhas)


def resumo_busca(nome, top=5):
    """Contagem de tentativas, tempo gasto, faixas, 5 melhores e configuração vencedora."""
    est = estudo(nome)
    estados = pd.Series([t.state.name for t in est.trials]).value_counts().to_dict()
    e = experimentos()
    e = e[(e["etapa"] == nome) & (e["status"] == "ok") & e["trial"].notna()]
    e = e.drop_duplicates("trial", keep="last").set_index("trial")
    tempo = experimentos().query("etapa == @nome")["tempo_treino_min"].sum()
    print(f"Busca {nome}: {len(est.trials)} tentativas {estados} | tempo de GPU registrado: {tempo / 60:.1f} h")
    linhas = []
    for t in est.trials:
        if t.state.name != "COMPLETE":
            continue
        r = e.loc[float(t.number)] if float(t.number) in e.index else None
        linhas.append({"tentativa": t.number, "bal_acc": t.value,
                       "spec95": r["val_spec95"] if r is not None else np.nan,
                       "AUC melanoma": r["val_mel_auc"] if r is not None else np.nan,
                       "AUC maligno": r["val_mal_auc"] if r is not None else np.nan,
                       "min": r["tempo_treino_min"] if r is not None else np.nan})
    melhores = pd.DataFrame(linhas).sort_values("bal_acc", ascending=False).head(top).round(4)
    vencedora = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in est.best_trial.params.items()}
    return _faixas(est), melhores, vencedora


def grafico_busca(nome, ax=None):
    """Valor de cada tentativa na ordem em que rodou (completas × cortadas) e o melhor até ali."""
    est = estudo(nome)
    ax = ax or plt.subplots(figsize=(6.4, 3.2))[1]
    comp = [(t.number, t.value) for t in est.trials if t.state.name == "COMPLETE"]
    cort = [(t.number, max(t.intermediate_values.values())) for t in est.trials
            if t.state.name == "PRUNED" and t.intermediate_values]
    if comp:
        x, y = zip(*comp)
        ax.scatter(x, y, s=42, color=AZUL, edgecolor=FUNDO, linewidth=1.5, zorder=3, label="completa (TTA)")
        melhor = np.maximum.accumulate(y)
        ax.step(x, melhor, where="post", color=AZUL, linewidth=1.2, alpha=0.6, label="melhor até ali")
    if cort:
        x, y = zip(*cort)
        ax.scatter(x, y, s=42, marker="X", color=LARANJA, edgecolor=FUNDO, linewidth=1, zorder=3,
                   label="cortada (última época vista)")
    ax.set_xlabel("tentativa (ordem)")
    ax.set_ylabel("acurácia balanceada (validação)")
    ax.set_title(f"{nome}: evolução da busca", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    return ax


def grafico_importancia(nome, ax=None):
    """Importância dos parâmetros (fANOVA do Optuna). Com poucas tentativas, é só indicativa."""
    import optuna
    est = estudo(nome)
    try:
        imp = optuna.importance.get_param_importances(est)
    except Exception as erro:  # noqa: BLE001
        print(f"importância indisponível para {nome}: {erro}")
        return None
    imp = pd.Series(imp).sort_values().tail(10)
    ax = ax or plt.subplots(figsize=(5.6, 0.32 * len(imp) + 0.8))[1]
    ax.barh(imp.index, imp.values, color=AZUL, height=0.6)
    for i, v in enumerate(imp.values):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=8, color=TEXTO2)
    ax.set_xlabel("importância relativa")
    ax.set_title(f"{nome}: o que mais pesou (fANOVA)", loc="left")
    ax.grid(axis="y", visible=False)
    return ax


def _prever(rid):
    z = np.load(achar(f"{rid}.npz", "previsoes"), allow_pickle=True)
    pred = ajustar(z["p"], float(z["tau"]), z["log_prior"]).argmax(1)
    return z["y"], pred


def recall_por_classe(ids, rotulos=None):
    """Acerto por classe (sensibilidade) e a confusão entre ceratose benigna, actínica e carc. espinocelular."""
    linhas = {}
    i3 = [CLASSES.index(c) for c in ["bkl", "ak", "scc"]]
    for rid, rot in zip(ids, rotulos or ids):
        y, pred = _prever(rid)
        cm = confusion_matrix(y, pred, labels=range(len(CLASSES)))
        m = cm[np.ix_(i3, i3)]
        linhas[rot] = list(cm.diagonal() / cm.sum(1)) + [(m.sum() - np.trace(m)) / cm[i3].sum()]
    return pd.DataFrame(linhas, index=CLASSES + ["confusão bkl/ak/scc"]).T.round(3)


def heatmap_recall(tab, ax=None):
    dados = tab[CLASSES]
    ax = ax or plt.subplots(figsize=(7.6, 0.45 * len(dados) + 1.2))[1]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ[1:])
    ax.imshow(dados.values, cmap=cmap, vmin=0.4, vmax=1.0, aspect="auto")
    for i in range(dados.shape[0]):
        for j in range(dados.shape[1]):
            v = dados.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if v > 0.78 else TEXTO)
    ax.set_xticks(range(len(CLASSES)), [NOMES[c] for c in CLASSES], rotation=30, ha="right")
    ax.set_yticks(range(len(dados)), dados.index)
    ax.grid(False)
    ax.set_title("Acerto por classe na validação (sensibilidade)", loc="left")
    return ax


def matriz_confusao(rid, titulo=None, ax=None):
    y, pred = _prever(rid)
    cm = confusion_matrix(y, pred, labels=range(len(CLASSES)), normalize="true")
    ax = ax or plt.subplots(figsize=(5.2, 4.6))[1]
    from matplotlib.colors import LinearSegmentedColormap
    ax.imshow(cm, cmap=LinearSegmentedColormap.from_list("seq", SEQ), vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j] >= 0.02:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if cm[i, j] > 0.55 else TEXTO)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("previsto")
    ax.set_ylabel("verdadeiro")
    ax.grid(False)
    ax.set_title(titulo or f"Matriz de confusão: {rid}", loc="left")
    return ax


def curva(rid, ax=None):
    h = pd.read_csv(achar(f"{rid}_historico.csv", "previsoes"))
    ax = ax or plt.subplots(figsize=(6.0, 3.0))[1]
    ax.plot(h["epoca"], h["val_bal_acc"], color=AZUL, linewidth=2, marker="o", markersize=5,
            markeredgecolor=FUNDO, label="acurácia balanceada")
    ax.plot(h["epoca"], h["val_mel_auc"], color=LARANJA, linewidth=2, marker="s", markersize=5,
            markeredgecolor=FUNDO, label="AUC melanoma")
    ax.set_xlabel("época")
    ax.set_ylabel("validação (sem TTA)")
    ax.set_title(f"{rid}: curva de treino", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    return ax


def tabela_finais(ids, rotulos):
    """Métricas de validação (TTA 8) dos modelos finais, lado a lado."""
    e = experimentos().set_index("id")
    cols = {"val_bal_acc": "bal_acc", "val_spec95": "spec95", "val_mel_auc": "AUC mel",
            "val_mal_auc": "AUC mal", "val_ece": "ECE", "tempo_treino_min": "treino (min)",
            "inferencia_ms_img": "ms/imagem"}
    linhas = []
    for rid, rot in zip(ids, rotulos):
        if rid not in e.index:
            continue
        r = e.loc[rid]
        r = r.iloc[-1] if isinstance(r, pd.DataFrame) else r
        cfg = json.loads(r["config"]) if isinstance(r["config"], str) else {}
        linhas.append({"modelo": rot, "res": cfg.get("res")} | {v: r[k] for k, v in cols.items()})
    return pd.DataFrame(linhas).set_index("modelo").round(4)


def grafico_finais(tab):
    """bal_acc e spec95 dos modelos finais (mesma escala 0–1, um eixo)."""
    # pontos (não barras): o eixo começa em 0,65 para mostrar diferenças pequenas, e barras com
    # eixo cortado exageram as diferenças
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(tab) + 1.0))
    y = np.arange(len(tab))
    for i, (a, s) in enumerate(zip(tab["bal_acc"], tab["spec95"])):
        ax.plot([min(a, s), max(a, s)], [i, i], color=GRADE, linewidth=2, zorder=1)
    ax.scatter(tab["bal_acc"], y, s=70, color=AZUL, edgecolor=FUNDO, linewidth=2, zorder=3,
               label="acurácia balanceada")
    ax.scatter(tab["spec95"], y, s=70, marker="D", color=LARANJA, edgecolor=FUNDO, linewidth=2, zorder=3,
               label="especificidade a 95% de sensib.")
    for i, (a, s) in enumerate(zip(tab["bal_acc"], tab["spec95"])):
        ax.text(max(a, s) + 0.004, i, f"{a:.3f} / {s:.3f}", va="center", fontsize=8, color=TEXTO2)
    ax.set_yticks(y, tab.index)
    ax.set_xlim(0.65, 0.83)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("validação (TTA 8)")
    ax.set_title("Modelos finais lado a lado", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    return ax

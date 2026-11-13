"""Treina uma configuração (uma tentativa) e registra o resultado em experimentos.csv.

Uso direto (E0, a receita atual portada do Keras):
    python -m dermato.treinar --etapa E0 [--rapido] [--seed 42] [--salvar]

As buscas (busca.py) chamam treinar(cfg, trial). Tudo que é escolhido (época, tau, limiar) vem
só da validação; os testes T1/T2 só são abertos por testar.py, nos finalistas.
"""
import argparse
import copy
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CLASSES, EXPERIMENTOS, RESULTADOS, SEED
from .dados import ler_splits, limitar_nevos, loader
from .metricas import avaliar, escolher_tau, limiar_sens, escore_maligno, softmax
from .modelos import MICRO_224, MICRO_FUSAO_224, Fusao, Rede, config_dados, congelar_bn

# Receita atual (modelo Keras do relatório), só com dermatoscopia e partindo do ImageNet
PADRAO = dict(
    etapa="E0", modelo="rede", backbone="effv2_b0", res=224, batch=32, epocas=8, paciencia=3,
    lr=1e-4, wd=0.02, aquecimento=1, perda="ce", gamma=2.0, suavizacao=0.0,
    aug="leve", mix_p=0.0, mix_tipo="mixup", sog=True, dropout=0.3, drop_path=0.0,
    bn_congelada=True, cabeca_epocas=0, meta=True, max_nevos=6000, balanceado=False, layer_decay=1.0,
    # fusão (E3/E4)
    juncao="concat", hidden=512, lr_vit_mult=1.0, ramos_congelados_epocas=0, pesos_cnn=None,
    emb=None, emb_dropout=0.0, emb_norm=False,
    seed=SEED, tta=8, salvar_modelo=False, rapido=False, workers=12,
)


def limitar_vram(fracao=0.90):
    """Teto de memória da GPU para o PyTorch: se uma configuração não couber, dá erro de memória na
    hora, em vez de o driver do Windows passar a usar a RAM do sistema (10x mais lento, sem aviso)."""
    torch.cuda.set_per_process_memory_fraction(fracao)


def fixar_sementes(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def construir(cfg):
    if cfg["modelo"] == "rede":
        m = Rede(cfg["backbone"], cfg["meta"], cfg["dropout"], cfg["drop_path"])
        mean, std = config_dados(m.backbone)
        micro = MICRO_224[cfg["backbone"]]
        backbones = {"backbone": (m.backbone, 1.0)}
    else:
        emb_dim = carregar_embeddings(cfg["emb"]).shape[1] if cfg["emb"] else 0
        m = Fusao(cfg["juncao"], cfg["hidden"], cfg["dropout"], cfg["drop_path"], cfg["meta"],
                  emb_dim, cfg["emb_dropout"], cfg["emb_norm"])
        if cfg["pesos_cnn"]:
            estado = torch.load(cfg["pesos_cnn"], map_location="cpu", weights_only=True)
            estado = {k.removeprefix("backbone."): v for k, v in estado.items() if k.startswith("backbone.")}
            m.cnn.load_state_dict(estado)
        mean, std = config_dados(m.cnn)
        micro = MICRO_FUSAO_224
        backbones = {"cnn": (m.cnn, 1.0), "vit": (m.vit, cfg["lr_vit_mult"])}
    micro = max(4, min(cfg["batch"], int(micro * (224 / cfg["res"]) ** 2)))
    return m, mean, std, micro, backbones


def grupos_otimizador(model, backbones, cfg):
    """Backbones com lr próprio (e decaimento por camada, se pedido); o resto (cabeça, metadados,
    junção) com o lr base. Sem weight decay em bias e normalizações."""
    grupos, vistos = [], set()
    for nome, (bb, mult) in backbones.items():
        congelado = cfg["cabeca_epocas"] if cfg["modelo"] == "rede" else cfg["ramos_congelados_epocas"]
        if cfg["layer_decay"] < 1.0:
            from timm.optim import param_groups_layer_decay
            gs = param_groups_layer_decay(bb, weight_decay=cfg["wd"], layer_decay=cfg["layer_decay"])
        else:
            dec = [p for p in bb.parameters() if p.ndim > 1]
            nodec = [p for p in bb.parameters() if p.ndim <= 1]
            gs = [{"params": dec, "weight_decay": cfg["wd"]}, {"params": nodec, "weight_decay": 0.0}]
        for g in gs:
            g["params"] = [p for p in g["params"] if p.requires_grad]
            g["lr_scale"] = g.get("lr_scale", 1.0) * mult
            g["congelado_epocas"] = congelado
            vistos.update(id(p) for p in g["params"])
            if g["params"]:
                grupos.append(g)
    resto = [p for p in model.parameters() if id(p) not in vistos and p.requires_grad]
    grupos.append({"params": [p for p in resto if p.ndim > 1], "weight_decay": cfg["wd"], "lr_scale": 1.0,
                   "congelado_epocas": 0})
    grupos.append({"params": [p for p in resto if p.ndim <= 1], "weight_decay": 0.0, "lr_scale": 1.0,
                   "congelado_epocas": 0})
    return grupos


def lr_no_passo(cfg, passo, por_epoca):
    """Aquecimento linear (lr/20 -> lr) e depois cosseno até 1% do lr, como no Keras original."""
    aq = cfg["aquecimento"] * por_epoca
    total = cfg["epocas"] * por_epoca
    if passo < aq:
        return cfg["lr"] / 20 + (cfg["lr"] - cfg["lr"] / 20) * passo / aq
    t = (passo - aq) / max(1, total - aq)
    return cfg["lr"] * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


def perda_fn(cfg, pesos_classe):
    def fn(logits, alvo):   # alvo: distribuição (N x C), já com suavização/mixup
        logp = F.log_softmax(logits.float(), 1)
        if cfg["perda"] == "focal":
            p = logp.exp()
            l = -(alvo * (1 - p).pow(cfg["gamma"]) * logp).sum(1)
        else:
            l = -(alvo * logp).sum(1)
        if cfg["perda"] == "ce_ponderada":
            l = l * (alvo @ pesos_classe)
        return l.mean()
    return fn


def misturar(x, alvo, cfg):
    if cfg["mix_p"] <= 0 or random.random() >= cfg["mix_p"]:
        return x, alvo
    perm = torch.randperm(x.size(0), device=x.device)
    if cfg["mix_tipo"] == "mixup":
        lam = float(np.random.beta(0.4, 0.4))
        x = lam * x + (1 - lam) * x[perm]
    else:   # cutmix
        lam = float(np.random.beta(1.0, 1.0))
        h, w = x.shape[-2:]
        rh, rw = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
        cy, cx = random.randint(0, h - 1), random.randint(0, w - 1)
        y0, y1, x0, x1 = max(cy - rh // 2, 0), min(cy + rh // 2, h), max(cx - rw // 2, 0), min(cx + rw // 2, w)
        x = x.clone()
        x[:, :, y0:y1, x0:x1] = x[perm, :, y0:y1, x0:x1]
        lam = 1 - (y1 - y0) * (x1 - x0) / (h * w)
    return x, lam * alvo + (1 - lam) * alvo[perm]


_EMB = {}


def carregar_embeddings(caminho):
    """Vetores do PanDerm alinhados com splits/dermato.csv (mesma ordem de linhas)."""
    if caminho not in _EMB:
        _EMB[caminho] = np.load(caminho, allow_pickle=True)["emb"].astype(np.float32)
    return _EMB[caminho]


@torch.no_grad()
def prever(model, dl, emb=None, tta=8):
    """Probabilidades médias de `tta` vistas (4 rotações x espelhamento)."""
    model.eval()
    saida = []
    for x, m, _, i in dl:
        x = x.cuda(non_blocking=True).to(memory_format=torch.channels_last)
        m = m.cuda(non_blocking=True)
        e = emb[i.numpy()].cuda() if emb is not None else None
        p = 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for k in range(tta):
                v = torch.rot90(x, k % 4, dims=(2, 3))
                if k >= 4:
                    v = torch.flip(v, dims=(3,))
                p = p + torch.softmax(model(v, m, e).float(), 1)
        saida.append((p / tta).cpu().numpy())
    return np.concatenate(saida)


def registrar(linha):
    EXPERIMENTOS.parent.mkdir(parents=True, exist_ok=True)
    novo = pd.DataFrame([linha])
    if EXPERIMENTOS.exists():
        antigo = pd.read_csv(EXPERIMENTOS)
        novo = pd.concat([antigo, novo], ignore_index=True)
    novo.to_csv(EXPERIMENTOS, index=False)


def treinar(cfg_parcial: dict, trial=None, nome=None):
    cfg = copy.deepcopy(PADRAO) | cfg_parcial
    limitar_vram()
    fixar_sementes(cfg["seed"])
    rid = nome or f"{cfg['etapa']}_{datetime.now():%Y%m%d_%H%M%S}" + (f"_t{trial.number:03d}" if trial else "")
    t0 = time.time()

    df = ler_splits()
    df["linha"] = np.arange(len(df))
    tr = limitar_nevos(df[df["split"] == "train"], cfg["max_nevos"], cfg["seed"])
    va = df[df["split"] == "val"].reset_index(drop=True)
    if cfg["rapido"]:   # teste de fumaça: poucas imagens, só as que já estão no cache
        existe = lambda d: d[[Path(c).exists() for c in d["caminho"]]]  # noqa: E731
        tr, va = existe(tr), existe(va)
        tr = tr.groupby("dx").head(40).reset_index(drop=True)
        va = va.groupby("dx").head(25).reset_index(drop=True)
        cfg["epocas"], cfg["tta"] = min(cfg["epocas"], 2), 2
    emb_all = torch.from_numpy(carregar_embeddings(cfg["emb"])) if cfg["emb"] else None
    emb_tr = emb_all[tr["linha"].to_numpy()] if emb_all is not None else None
    emb_va = emb_all[va["linha"].to_numpy()] if emb_all is not None else None

    model, mean, std, micro, backbones = construir(cfg)
    model = model.cuda().to(memory_format=torch.channels_last)
    acum = math.ceil(cfg["batch"] / micro)
    micro = cfg["batch"] // acum

    cont = tr["y"].value_counts().reindex(range(len(CLASSES))).fillna(1).to_numpy()
    uniforme = cfg["balanceado"] or cfg["perda"] == "ce_ponderada"
    prior = np.full(len(CLASSES), 1 / len(CLASSES)) if uniforme else cont / cont.sum()
    log_prior = np.log(prior)
    pesos_classe = torch.tensor(cont.sum() / (len(CLASSES) * cont), dtype=torch.float32).cuda()
    if cfg["balanceado"]:
        pesos_classe = torch.ones_like(pesos_classe)

    kw = dict(mean=mean, std=std, workers=cfg["workers"], seed=cfg["seed"], sog=cfg["sog"])
    dl_tr = loader(tr, cfg["res"], True, micro, cfg["aug"], balanceado=cfg["balanceado"], **kw)
    dl_va = loader(va, cfg["res"], False, micro * 2, **(kw | {"workers": 4}))

    opt = torch.optim.AdamW(grupos_otimizador(model, backbones, cfg), lr=cfg["lr"])
    perda = perda_fn(cfg, pesos_classe)
    por_epoca = len(dl_tr) // acum
    y_va = va["y"].to_numpy()

    melhor, melhor_estado, melhor_epoca, sem_melhora, historico = -1.0, None, -1, 0, []
    passo = 0
    status = "ok"
    try:
        for ep in range(cfg["epocas"]):
            model.train()
            if cfg["bn_congelada"]:
                congelar_bn(model)
            soma, n = 0.0, 0
            opt.zero_grad(set_to_none=True)
            for k, (x, m, y, i) in enumerate(dl_tr):
                x = x.cuda(non_blocking=True).to(memory_format=torch.channels_last)
                m, y = m.cuda(non_blocking=True), y.cuda(non_blocking=True)
                e = emb_tr[i.numpy()].cuda() if emb_tr is not None else None
                alvo = F.one_hot(y, len(CLASSES)).float()
                if cfg["suavizacao"]:
                    alvo = alvo * (1 - cfg["suavizacao"]) + cfg["suavizacao"] / len(CLASSES)
                x, alvo = misturar(x, alvo, cfg)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    l = perda(model(x, m, e), alvo) / acum
                l.backward()
                soma += float(l.detach()) * acum
                n += 1
                if (k + 1) % acum == 0:
                    lr = lr_no_passo(cfg, passo, por_epoca)
                    for g in opt.param_groups:
                        g["lr"] = 0.0 if ep < g["congelado_epocas"] else lr * g["lr_scale"]
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    passo += 1
            p_va = prever(model, dl_va, emb_va, tta=1)
            tau_ep = escolher_tau(p_va, y_va, log_prior)
            r = avaliar(p_va, y_va, tau_ep, log_prior)
            historico.append({"epoca": ep + 1, "perda_treino": soma / max(n, 1), "val_bal_acc": r["bal_acc"],
                              "val_mel_auc": r["mel_auc"], "tau": tau_ep})
            print(f"  [{rid}] época {ep + 1}/{cfg['epocas']}: perda {soma / max(n, 1):.4f} | val bal_acc "
                  f"{r['bal_acc']:.4f} mel_auc {r['mel_auc']:.4f} spec95 {r['spec95']:.4f} (tau {tau_ep}) "
                  f"| {(time.time() - t0) / 60:.1f} min", flush=True)
            if r["bal_acc"] > melhor:
                melhor, melhor_epoca, sem_melhora = r["bal_acc"], ep + 1, 0
                melhor_estado = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
            else:
                sem_melhora += 1
            if trial is not None:
                import optuna
                trial.report(r["bal_acc"], ep)
                if trial.should_prune():
                    status = "cortada"
                    raise optuna.TrialPruned()
            if sem_melhora >= cfg["paciencia"]:
                break
    except BaseException as erro:
        if status == "ok":
            status = f"erro: {type(erro).__name__}"
        raise
    finally:
        tempo_treino = time.time() - t0
        linha = {"id": rid, "etapa": cfg["etapa"], "familia": cfg["backbone"] if cfg["modelo"] == "rede" else
                 f"fusao_{cfg['juncao']}" + ("+panderm" if cfg["emb"] else ""),
                 "trial": trial.number if trial else None, "seed": cfg["seed"], "status": status,
                 "melhor_epoca": melhor_epoca, "epocas_rodadas": len(historico),
                 "tempo_treino_min": round(tempo_treino / 60, 2), "val_bal_acc_epoca": melhor,
                 "config": json.dumps({k: v for k, v in cfg.items() if k not in ("workers",)}, default=str)}
        if status != "ok":
            registrar(linha)
            del dl_tr, dl_va, model, opt
            torch.cuda.empty_cache()

    # --- melhor época, avaliação final na validação com TTA
    model.load_state_dict(melhor_estado)
    t_inf = time.time()
    p_va = prever(model, dl_va, emb_va, tta=cfg["tta"])
    ms_img = (time.time() - t_inf) / len(va) / cfg["tta"] * 1000
    tau = escolher_tau(p_va, y_va, log_prior)
    mal = np.isin(y_va, [CLASSES.index(c) for c in ["ak", "bcc", "mel", "scc"]])
    limiar = limiar_sens(mal, escore_maligno(p_va))
    r = avaliar(p_va, y_va, tau, log_prior, limiar=limiar, prefixo="val_")
    linha |= {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}
    linha |= {"limiar": round(limiar, 5), "inferencia_ms_img": round(ms_img, 3)}

    pasta = RESULTADOS / "previsoes"
    pasta.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pasta / f"{rid}.npz", isic_id=va["isic_id"].to_numpy(), p=p_va, y=y_va,
                        log_prior=log_prior, tau=tau, limiar=limiar)
    pd.DataFrame(historico).to_csv(pasta / f"{rid}_historico.csv", index=False)
    if cfg["salvar_modelo"]:
        (RESULTADOS / "modelos").mkdir(parents=True, exist_ok=True)
        torch.save(melhor_estado, RESULTADOS / "modelos" / f"{rid}.pt")
        linha["modelo"] = str(RESULTADOS / "modelos" / f"{rid}.pt")
    registrar(linha)
    print(f"  [{rid}] FIM val: bal_acc {r['val_bal_acc']:.4f} | spec95 {r['val_spec95']:.4f} | "
          f"mel_auc {r['val_mel_auc']:.4f} | mal_auc {r['val_mal_auc']:.4f} | ece {r['val_ece']:.4f} | "
          f"tau {tau} | {tempo_treino / 60:.1f} min", flush=True)
    del dl_tr, dl_va, model, opt
    torch.cuda.empty_cache()
    return linha


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapa", default="E0")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--salvar", action="store_true")
    ap.add_argument("--nome", help="id da rodada (padrão: etapa + data)")
    ap.add_argument("--cfg", default="{}", help="JSON com parâmetros que substituem o padrão")
    a = ap.parse_args()
    treinar({"etapa": a.etapa, "seed": a.seed, "rapido": a.rapido, "salvar_modelo": a.salvar} | json.loads(a.cfg),
            nome=a.nome)

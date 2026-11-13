"""Vetores congelados do PanDerm (ViT-L/16, Yan et al., 2025) e o teste rápido antes da E4.

Uso:
    python -m dermato.panderm extrair       # GPU, alguns minutos: um vetor de 1024 por imagem
    python -m dermato.panderm rapido        # CPU: regressão logística, MLP e LightGBM nos vetores

Os vetores ficam em RESULTADOS/panderm_emb.npz, na mesma ordem de linhas de splits/dermato.csv
(é o que a E4 usa como terceiro ramo). O pré-processamento é o do repositório oficial:
Resize(256) + CenterCrop(224), normalização ImageNet, sem Shades of Gray.
"""
import sys
import time

import numpy as np
import pandas as pd

from .config import CLASSES, DATA, MAL_IDX, RESULTADOS, SEED

PASTA = DATA / "panderm"
CKPT = PASTA / "panderm_ll_data6_checkpoint-499.pth"
EMB = RESULTADOS / "panderm_emb.npz"


def carregar_panderm():
    """Encoder do PanDerm (estilo CAE) nos pesos de um ViT-L/16 do timm.

    Só lemos o arquivo de pesos (weights_only), sem executar o código do repositório deles.
    Mapeamento: gamma_1/2 -> ls1/ls2 (layer scale); q_bias e v_bias -> qkv.bias = [q, 0, v]
    (a chave não tem bias, como no BEiT/CAE). O vetor é o token [CLS] depois da norma final,
    igual ao forward_features do PanDerm em modo linear probing.
    """
    import timm
    import torch
    bruto = torch.load(CKPT, map_location="cpu", weights_only=True)
    enc = {k.removeprefix("encoder."): v for k, v in bruto.items() if k.startswith("encoder.")}
    estado = {}
    for k, v in enc.items():
        if k.endswith("attn.q_bias"):
            pre = k.removesuffix("q_bias")
            estado[pre + "qkv.bias"] = torch.cat([v, torch.zeros_like(v), enc[pre + "v_bias"]])
        elif k.endswith("attn.v_bias"):
            continue
        else:
            estado[k.replace("gamma_1", "ls1.gamma").replace("gamma_2", "ls2.gamma")] = v
    model = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0, init_values=1e-5)
    model.load_state_dict(estado, strict=True)   # falha se sobrar ou faltar qualquer peso
    print(f"PanDerm: {len(estado)} tensores do encoder carregados no ViT-L/16 (strict)", flush=True)
    return model.eval()


class _DSPanDerm:
    """Dataset do PanDerm no nível do módulo: no Windows os processos de carregamento (spawn)
    precisam conseguir importar a classe (uma classe local dentro de extrair() quebrava)."""
    def __init__(self, caminhos):
        from torchvision import transforms
        self.caminhos = caminhos
        self.tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.228, 0.224, 0.225))])

    def __len__(self):
        return len(self.caminhos)

    def __getitem__(self, i):
        from PIL import Image
        return self.tf(Image.open(self.caminhos[i]).convert("RGB"))


def extrair():
    import torch
    from torch.utils.data import DataLoader

    from .dados import ler_splits
    df = ler_splits()
    model = carregar_panderm().cuda()
    dl = DataLoader(_DSPanDerm(df["caminho"].tolist()), batch_size=128, num_workers=10, pin_memory=True)
    saida, t0 = [], time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for k, x in enumerate(dl):
            saida.append(model(x.cuda(non_blocking=True)).float().cpu().numpy())
            if k % 50 == 0:
                print(f"  {k * 128}/{len(df)} | {(k + 1) * 128 / (time.time() - t0):.0f} img/s", flush=True)
    emb = np.concatenate(saida)
    EMB.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EMB, emb=emb.astype(np.float16), isic_id=df["isic_id"].to_numpy())
    print(f"-> {EMB} {emb.shape} em {(time.time() - t0) / 60:.1f} min")


def rapido():
    """Classificadores simples sobre os vetores, treinados no treino e medidos na validação."""
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    from .dados import ler_splits, meta_features
    from .metricas import avaliar, escolher_tau, escore_maligno, limiar_sens
    from .treinar import registrar

    df = ler_splits()
    z = np.load(EMB, allow_pickle=True)   # isic_id é texto (array de objetos)
    assert (z["isic_id"] == df["isic_id"].to_numpy()).all()
    X = np.concatenate([z["emb"].astype(np.float32), meta_features(df)], 1)
    tr, va = (df["split"] == "train").to_numpy(), (df["split"] == "val").to_numpy()
    y = df["y"].to_numpy()
    esc = StandardScaler().fit(X[tr])
    Xtr, Xva = esc.transform(X[tr]), esc.transform(X[va])
    cont = np.bincount(y[tr], minlength=len(CLASSES))
    log_prior = np.log(np.full(len(CLASSES), 1 / len(CLASSES)))   # todos com peso balanceado

    modelos = {
        "regressao_logistica": LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced"),
        "mlp": MLPClassifier((512,), alpha=1e-3, early_stopping=True, max_iter=200, random_state=SEED),
        "lightgbm": lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8,
                                       colsample_bytree=0.5, class_weight="balanced", random_state=SEED, verbose=-1),
    }
    linhas = []
    for nome, m in modelos.items():
        t0 = time.time()
        if nome == "mlp":   # MLP do sklearn não tem class_weight: reamostra balanceado
            rng = np.random.default_rng(SEED)
            idx = np.concatenate([rng.choice(np.where(y[tr] == c)[0], max(cont), replace=True)
                                  for c in range(len(CLASSES))])
            m.fit(Xtr[idx], y[tr][idx])
        else:
            m.fit(Xtr, y[tr])
        p = m.predict_proba(Xva)
        tau = escolher_tau(p, y[va], log_prior)
        mal = np.isin(y[va], MAL_IDX)
        lim = limiar_sens(mal, escore_maligno(p))
        r = avaliar(p, y[va], tau, log_prior, limiar=lim, prefixo="val_")
        rid = f"PanDerm_rapido_{nome}"
        np.savez_compressed(RESULTADOS / "previsoes" / f"{rid}.npz", isic_id=df["isic_id"][va].to_numpy(), p=p,
                            y=y[va], log_prior=log_prior, tau=tau, limiar=lim)
        linha = {"id": rid, "etapa": "PanDerm_rapido", "familia": nome, "seed": SEED, "status": "ok",
                 "tempo_treino_min": round((time.time() - t0) / 60, 2), "limiar": lim} | r
        registrar(linha)
        linhas.append(linha)
        print(f"{nome:20s} val bal_acc {r['val_bal_acc']:.4f} | spec95 {r['val_spec95']:.4f} | "
              f"mel_auc {r['val_mel_auc']:.4f} | mal_auc {r['val_mal_auc']:.4f} | {linha['tempo_treino_min']} min", flush=True)
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    (RESULTADOS / "previsoes").mkdir(parents=True, exist_ok=True)
    {"extrair": extrair, "rapido": rapido}[sys.argv[1]]()

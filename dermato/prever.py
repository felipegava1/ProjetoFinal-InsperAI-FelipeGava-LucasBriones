"""Roda um modelo treinado em imagens de dermatoscopia novas.

Uso:
    python -m dermato.prever IMAGEM_OU_PASTA [...] [--modelo E1_final_s42] [--meta metadados.csv]
                             [--tta 8] [--cpu] [--saida previsoes.csv]

Precisa só de: o repositório, as dependências (requirements.txt) e o arquivo de pesos
resultados/modelos/<modelo>.pt. Não precisa das bases de dados.

Saída (uma linha por imagem, também gravada em CSV):
  classe       classe mais provável depois do ajuste de prior (tau escolhido na validação)
  p_<classe>   probabilidades do modelo para as 8 classes (sem ajuste)
  escore_maligno  soma de p(ak, bcc, mel, scc)
  triagem      "prioritaria" se escore_maligno >= limiar (95% de sensibilidade na validação)

--meta: CSV opcional com colunas arquivo (nome do arquivo), idade, sexo (male/female) e local
(uma das 8 regiões do ISIC 2019: anterior torso, head/neck, lateral torso, lower extremity,
oral/genital, palms/soles, posterior torso, upper extremity). Sem ele, os metadados entram como
faltantes, situação que o modelo também viu no treino.

ATENÇÃO: projeto acadêmico. O resultado é apoio à priorização, nunca diagnóstico; o diagnóstico
de câncer de pele é a histopatologia.
"""
import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.io import ImageReadMode, decode_jpeg
from torchvision.transforms import v2 as T

from .config import CACHE_LADO, CLASSES, EXPERIMENTOS, MAL_IDX, achar
from .dados import RecorteCentral, meta_features, shades_of_gray
from .metricas import ajustar

NOMES = {"ak": "ceratose actínica", "bcc": "carcinoma basocelular", "bkl": "ceratose benigna",
         "df": "dermatofibroma", "mel": "melanoma", "nv": "nevo", "scc": "carcinoma espinocelular",
         "vasc": "lesão vascular"}
EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def carregar_modelo(rid, dispositivo):
    from .treinar import PADRAO, construir
    exp = pd.read_csv(EXPERIMENTOS)
    linhas = exp[exp["id"] == rid]
    if linhas.empty:
        raise SystemExit(f"Modelo {rid} não está em {EXPERIMENTOS}. Disponíveis: "
                         f"{', '.join(i for i in exp['id'] if str(i).endswith('_final_s42'))}")
    cfg = PADRAO | json.loads(linhas.iloc[-1]["config"])
    if cfg.get("emb"):
        raise SystemExit("Modelos com vetores do PanDerm (E4) ainda não são suportados aqui.")
    model, mean, std, _, _ = construir(cfg | {"pesos_cnn": None, "pretrained": False})
    pesos = achar(f"{rid}.pt", "modelos")
    if pesos.stat().st_size < 1024:   # só o ponteiro do Git LFS, sem o arquivo de verdade
        raise SystemExit(f"{pesos} é um ponteiro do Git LFS ({pesos.stat().st_size} bytes), não os pesos. "
                         "Rode: git lfs install && git lfs pull")
    estado = torch.load(pesos, map_location="cpu", weights_only=True)
    model.load_state_dict({k: v.float() for k, v in estado.items()})
    calib = np.load(achar(f"{rid}.npz", "previsoes"), allow_pickle=True)
    return model.to(dispositivo).eval(), cfg, mean, std, calib


def preparar(caminho, cfg, mean, std):
    """Igual à avaliação do treino: menor lado 448 (como o cache), Shades of Gray se o modelo usou,
    quadrado central e redimensionamento para a resolução do modelo."""
    # mesmo caminho do cache do treino (preparar._redimensionar): redução por draft do JPEG + LANCZOS,
    # regravação em JPEG 95 e decodificação pelo torchvision
    im = Image.open(caminho)
    escala = min(1.0, CACHE_LADO / min(im.size))
    novo = (max(1, round(im.width * escala)), max(1, round(im.height * escala)))
    if im.format == "JPEG":
        im.draft("RGB", novo)
    im = im.convert("RGB")
    if im.size != novo:
        im = im.resize(novo, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=95)
    x = decode_jpeg(torch.frombuffer(bytearray(buf.getvalue()), dtype=torch.uint8), mode=ImageReadMode.RGB)
    if cfg["sog"]:
        x = shades_of_gray(x)
    tf = T.Compose([RecorteCentral(cfg["res"]), T.ToDtype(torch.float32, scale=True), T.Normalize(mean, std)])
    return tf(x)


def listar(entradas):
    arquivos = []
    for e in map(Path, entradas):
        if e.is_dir():
            arquivos += sorted(p for p in e.rglob("*") if p.suffix.lower() in EXTENSOES)
        elif e.suffix.lower() in EXTENSOES:
            arquivos.append(e)
        else:
            raise SystemExit(f"Não é imagem nem pasta: {e}")
    if not arquivos:
        raise SystemExit("Nenhuma imagem encontrada.")
    return arquivos


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("entradas", nargs="+", help="imagens ou pastas")
    ap.add_argument("--modelo", default="E1_final_s42")
    ap.add_argument("--meta", help="CSV com arquivo, idade, sexo, local")
    ap.add_argument("--tta", type=int, default=8, help="vistas (rotações x espelhamento); 1 = mais rápido")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--saida", default="previsoes.csv")
    a = ap.parse_args()

    disp = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    model, cfg, mean, std, calib = carregar_modelo(a.modelo, disp)
    arquivos = listar(a.entradas)

    meta = pd.DataFrame({"arquivo": [p.name for p in arquivos], "idade": np.nan, "sexo": None, "local": None})
    if a.meta:
        m = pd.read_csv(a.meta).set_index("arquivo")
        for c in ("idade", "sexo", "local"):
            if c in m:
                meta[c] = meta["arquivo"].map(m[c])
    meta["idade"] = pd.to_numeric(meta["idade"], errors="coerce")
    M = torch.from_numpy(meta_features(meta))

    probs = []
    with torch.no_grad():
        for i in range(0, len(arquivos), a.lote):
            x = torch.stack([preparar(p, cfg, mean, std) for p in arquivos[i:i + a.lote]]).to(disp)
            m = M[i:i + a.lote].to(disp)
            p = 0
            for k in range(a.tta):
                v = torch.rot90(x, k % 4, dims=(2, 3))
                if k >= 4:
                    v = torch.flip(v, dims=(3,))
                p = p + torch.softmax(model(v, m).float(), 1)
            probs.append((p / a.tta).cpu().numpy())
            print(f"  {min(i + a.lote, len(arquivos))}/{len(arquivos)}", flush=True)
    p = np.concatenate(probs)

    tau, limiar, log_prior = float(calib["tau"]), float(calib["limiar"]), calib["log_prior"]
    pred = ajustar(p, tau, log_prior).argmax(1)
    escore = p[:, MAL_IDX].sum(1)
    out = pd.DataFrame({"arquivo": [str(x) for x in arquivos], "classe": [CLASSES[i] for i in pred],
                        "nome_classe": [NOMES[CLASSES[i]] for i in pred],
                        "escore_maligno": escore.round(4),
                        "triagem": np.where(escore >= limiar, "prioritaria", "rotina")})
    for j, c in enumerate(CLASSES):
        out[f"p_{c}"] = p[:, j].round(4)
    out.to_csv(a.saida, index=False)
    pd.set_option("display.width", 200)
    print(out[["arquivo", "nome_classe", "escore_maligno", "triagem"]].to_string(index=False))
    print(f"\nModelo {a.modelo} ({cfg['res']} px, tau {tau}, limiar de triagem {limiar:.3f}) -> {a.saida}")
    print("Apoio à priorização, não diagnóstico.")


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.12"
# dependencies = ["tensorflow-cpu", "pandas", "pillow", "numpy", "scikit-learn"]
# ///
"""Testa o modelo de pintas (modelos/pintas/melhor.keras, EfficientNetV2-B0 em 224 px) nas bases
que baixamos (data/processed/manifesto.csv).

Uso:  uv run scripts/testar_modelo.py CAMINHO/melhor.keras [opções]
        --grupos G [G ...]   só alguns grupos (ver GRUPOS abaixo)
        --amostra N          no máximo N imagens por grupo (sorteio fixo), para um teste rápido
        --tta                média de 8 visões (4 rotações x espelhamento), como no notebook; 8x mais lento
        --tau T              intensidade do ajuste de prior (padrão: 1,1, o escolhido na validação)
        --limiar L           limiar da triagem maligno x benigno (padrão: 0,287, o da validação)

O pré-processamento é o mesmo do pipeline.py (Shades of Gray, recorte central, metadados), para
que a avaliação valha pelo modelo e não pela entrada. As imagens vêm das cópias de 256 px.

Grupos disponíveis (atenção: isic2020, milk10k, mskcc, anonima e mednode entram no treino do modelo;
só isic2019_test, pad_ufes_20 e hiba ficam inteiramente de fora):
  isic2019_test             dermatoscopia, 6.191 rotuladas (inclui as 1.511 do teste 2018)
  isic2020                  dermatoscopia, 33.126 (584 melanomas; a maioria "benigno sem diagnóstico",
                            então só as métricas binárias valem)
  milk10k_dermatoscopia     dermatoscopia, 5.240 (450 melanomas)
  milk10k_foto              foto clínica de perto da mesma lesão, 5.240
  pad_ufes_20               foto de celular, 2.298 (52 melanomas)
  hiba                      foto de celular, 355 (65 melanomas)
  mskcc, anonima            fotos clínicas de câmera, 1.276 e 100
  mednode                   foto clínica de câmera, 170 (70 melanomas)

Saída em data/processed/avaliacao_professor/:
  previsoes.csv            probabilidades das 8 classes por imagem
  resumo_por_grupo.csv     métricas por grupo
  melanoma_por_grupo.csv   em que classe caem os melanomas de cada grupo
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

RAIZ = Path(__file__).resolve().parent.parent
PROC = RAIZ / "data" / "processed"
SAIDA = PROC / "avaliacao_professor"
sys.path.insert(0, str(RAIZ))  # prepare_data.py e pipeline.py, na raiz do repositório

from prepare_data import CLASSES  # noqa: E402  ["ak","bcc","bkl","df","mel","nv","scc","vasc"]
from pipeline import SITES, meta_features, shades_of_gray  # noqa: E402

IMG_SIZE = 224
# Contagem das classes no treino do modelo: define o prior que o ajuste de logits desconta
# na hora de decidir.
TREINO = {"ak": 910, "bcc": 6384, "bkl": 3145, "df": 275, "mel": 4981, "nv": 6000, "scc": 1460, "vasc": 280}
LOG_PRIOR = np.log(np.array([TREINO[c] for c in CLASSES]) / sum(TREINO.values()))
MALIGNAS = ["ak", "bcc", "mel", "scc"]  # ak conta como maligna, como o akiec do HAM10000
IDX = {c: i for i, c in enumerate(CLASSES)}

# nossa classe harmonizada -> classe do modelo (as outras ficam só nas métricas binárias)
PARA_8_CLASSES = {"MEL": "mel", "NV": "nv", "BCC": "bcc", "SCC": "scc", "AK": "ak", "BKL": "bkl", "DF": "df", "VASC": "vasc"}
# nossa região anatômica -> as 8 regiões do ISIC 2019 que o modelo conhece. "Tronco" sem dizer se
# é frente ou costas fica como faltante, como no HAM10000.
PARA_SITE = {s: s for s in SITES} | {
    "Head and neck": "head/neck", "head_neck_face": "head/neck",
    "Upper extremity": "upper extremity", "upper_extremity": "upper extremity", "hand": "upper extremity",
    "Lower extremity": "lower extremity", "lower_extremity": "lower extremity", "foot": "lower extremity",
    "genital": "oral/genital",
}


def ler_manifesto():
    """manifesto.csv gerado por preparar_dados.py ou, se não houver, o comprimido do repositório."""
    arquivo = PROC / "manifesto.csv"
    return pd.read_csv(arquivo if arquivo.exists() else PROC / "manifesto.csv.gz", low_memory=False)


def definir_grupos(m):
    """Nome do grupo de avaliação de cada linha do manifesto (NaN = fora da avaliação)."""
    g = pd.Series(np.nan, index=m.index, dtype=object)
    g[(m["base"] == "isic2019") & (m["particao_oficial"] == "test")] = "isic2019_test"
    g[m["base"] == "isic2020"] = "isic2020"
    g[(m["base"] == "milk10k") & (m["tipo_imagem"] == "dermatoscopia")] = "milk10k_dermatoscopia"
    g[(m["base"] == "milk10k") & (m["tipo_imagem"] != "dermatoscopia")] = "milk10k_foto"
    fontes = {"PAD-UFES-20": "pad_ufes_20", "Hospital Italiano de Buenos Aires": "hiba",
              "Memorial Sloan Kettering Cancer Center": "mskcc", "Anonymous": "anonima"}
    cl = m["base"] == "isic_clinicas"
    g[cl] = m.loc[cl, "fonte"].map(fontes)
    g[m["base"] == "mednode"] = "mednode"
    return g


GRUPOS = ["isic2019_test", "isic2020", "milk10k_dermatoscopia", "milk10k_foto",
          "pad_ufes_20", "hiba", "mskcc", "anonima", "mednode"]


def carregar_imagem(arquivo):
    """Igual ao pipeline.py na avaliação: Shades of Gray, recorte central quadrado, 224 px, 0-255."""
    im = Image.open(PROC / arquivo).convert("RGB")
    im = Image.fromarray(shades_of_gray(np.asarray(im)))
    lado = min(im.size)
    x0, y0 = (im.width - lado) // 2, (im.height - lado) // 2
    im = im.crop((x0, y0, x0 + lado, y0 + lado)).resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ajustar(p, tau):
    return softmax(np.log(p + 1e-12) - tau * LOG_PRIOR)


def visoes(x, k):
    x = np.rot90(x, k % 4, axes=(1, 2))
    return x[:, :, ::-1] if k >= 4 else x


def prever(modelo, df, usa_meta, tta, lote=128):
    meta = meta_features(df) if usa_meta else None
    saida = np.zeros((len(df), len(CLASSES)), dtype=np.float32)
    inicio = time.time()
    with ThreadPoolExecutor(8) as ex:
        for i in range(0, len(df), lote):
            x = np.stack(list(ex.map(carregar_imagem, df["arquivo"].iloc[i:i + lote])))
            p = 0
            for k in range(8 if tta else 1):
                entrada = {"image": visoes(x, k), "meta": meta[i:i + lote]} if usa_meta else visoes(x, k)
                p = p + softmax(modelo.predict(entrada, verbose=0))
            saida[i:i + lote] = p / (8 if tta else 1)
            feitas = min(i + lote, len(df))
            print(f"\r  {feitas}/{len(df)}  {feitas / (time.time() - inicio):.0f} img/s", end="", flush=True)
    print()
    return saida


def avaliar_grupo(g, tau, limiar):
    p = g[[f"p_{c}" for c in CLASSES]].to_numpy()
    p_adj = ajustar(p, tau)
    pred = p_adj.argmax(1)
    linha = {"grupo": g["grupo"].iloc[0], "imagens": len(g), "melanomas": int((g["classe"] == "MEL").sum()),
             "tipo": g["tipo_imagem"].mode().iloc[0]}

    # 8 classes: só as imagens cuja classe o modelo conhece
    oito = g["classe_8"].notna().to_numpy()
    if oito.sum() >= 10:
        y8 = g.loc[oito, "classe_8"].map(IDX).to_numpy()
        linha["imagens_8_classes"] = int(oito.sum())
        linha["classes_presentes"] = int(len(set(y8)))
        linha["acuracia"] = accuracy_score(y8, pred[oito])
        linha["acuracia_balanceada"] = balanced_accuracy_score(y8, pred[oito])

    # melanoma: sensibilidade e precisão da decisão (classe prevista = mel) e AUC do p_mel
    mel = (g["classe"] == "MEL").to_numpy()
    e_mel = pred == IDX["mel"]
    if mel.sum() > 0:
        linha["mel_sensibilidade"] = float(e_mel[mel].mean())
        linha["mel_precisao"] = float(mel[e_mel].mean()) if e_mel.any() else np.nan
        if (~mel).sum() > 0:
            linha["mel_auc"] = roc_auc_score(mel, p[:, IDX["mel"]])
    linha["mel_falsos_positivos_%"] = 100 * float(e_mel[~mel].mean()) if (~mel).sum() else np.nan

    # triagem maligno x benigno com o limiar dado (soma de ak, bcc, mel, scc, sem ajuste)
    escore = p[:, [IDX[c] for c in MALIGNAS]].sum(1)
    mal = g["maligno_8"].to_numpy()
    valido = ~np.isnan(mal)
    if valido.sum() and (mal[valido] == 1).sum() and (mal[valido] == 0).sum():
        mal_b = mal[valido] == 1
        linha["maligno_auc"] = roc_auc_score(mal_b, escore[valido])
        linha["triagem_sensibilidade"] = float((escore[valido][mal_b] >= limiar).mean())
        linha["triagem_especificidade"] = float((escore[valido][~mal_b] < limiar).mean())
    return linha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modelo")
    ap.add_argument("--grupos", nargs="+", default=GRUPOS, choices=GRUPOS)
    ap.add_argument("--amostra", type=int)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--tau", type=float, default=1.1)
    ap.add_argument("--limiar", type=float, default=0.287)
    args = ap.parse_args()

    import keras  # depois do argparse, para o --help ser instantâneo
    modelo = keras.models.load_model(args.modelo, compile=False)
    entradas = modelo.inputs if isinstance(modelo.inputs, list) else [modelo.inputs]
    usa_meta = len(entradas) > 1
    print(f"modelo {modelo.name}: {modelo.count_params() / 1e6:.1f} M parâmetros, "
          f"{'imagem + metadados' if usa_meta else 'só imagem'}")

    m = ler_manifesto()
    m["grupo"] = definir_grupos(m)
    m = m[m["grupo"].isin(args.grupos) & m["arquivo"].notna() & ~m["classe"].isin(["UNK", "INDETERMINADO"])]
    m = m[m["duplicada_de"].isna()] if "duplicada_de" in m else m
    if args.amostra:
        m = pd.concat([d.sample(min(len(d), args.amostra), random_state=0) for _, d in m.groupby("grupo")])
    m = m.sort_values(["grupo", "isic_id"]).reset_index(drop=True)

    m["classe_8"] = m["classe"].map(PARA_8_CLASSES)
    m["maligno_8"] = m["classe"].map(
        {**{c: 1.0 for c in ["MEL", "BCC", "SCC", "AK", "OUTRO_MALIGNO"]},
         **{c: 0.0 for c in ["NV", "BKL", "DF", "VASC", "INF", "OUTRO_BENIGNO", "BENIGNO_SEM_DIAGNOSTICO"]}})
    # metadados nos nomes que o pipeline.py espera
    m["age_approx"] = m["idade"]
    m["sex"] = m["sexo"].where(m["sexo"].isin(["male", "female"]))
    m["site"] = m["local_anatomico"].map(PARA_SITE)
    print("imagens por grupo:", m["grupo"].value_counts().to_dict())
    print("regiões reconhecidas:", f"{m['site'].notna().mean():.0%}", "| idade:", f"{m['age_approx'].notna().mean():.0%}")

    probs = prever(modelo, m, usa_meta, args.tta)
    for i, c in enumerate(CLASSES):
        m[f"p_{c}"] = probs[:, i]
    m["previsto"] = [CLASSES[i] for i in ajustar(probs, args.tau).argmax(1)]

    SAIDA.mkdir(parents=True, exist_ok=True)
    cols = ["isic_id", "grupo", "base", "fonte", "tipo_imagem", "classe", "classe_8", "previsto"] + [f"p_{c}" for c in CLASSES]
    m[cols].to_csv(SAIDA / "previsoes.csv", index=False)

    resumo = pd.DataFrame([avaliar_grupo(g, args.tau, args.limiar) for _, g in m.groupby("grupo", sort=False)])
    resumo = resumo.set_index("grupo").loc[[g for g in GRUPOS if g in set(m["grupo"])]].reset_index()
    resumo.to_csv(SAIDA / "resumo_por_grupo.csv", index=False)

    mel = m[m["classe"] == "MEL"]
    destino = pd.crosstab(mel["grupo"], mel["previsto"], normalize="index").mul(100).round(1)
    destino = destino.reindex(columns=CLASSES, fill_value=0).loc[[g for g in GRUPOS if g in destino.index]]
    destino.to_csv(SAIDA / "melanoma_por_grupo.csv")

    pd.set_option("display.width", 250)
    print(f"\n=== Métricas por grupo (tau={args.tau}, limiar={args.limiar}, {'com' if args.tta else 'sem'} TTA) ===")
    print(resumo.round(3).to_string(index=False))
    print("\n=== Para onde vão os melanomas (% dos melanomas de cada grupo, por classe prevista) ===")
    print(destino.to_string())
    print(f"\nArquivos em {SAIDA}")


if __name__ == "__main__":
    main()

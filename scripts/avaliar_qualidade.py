# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "pillow", "numpy"]
# ///
"""Avalia a qualidade das imagens já preparadas (data/processed/manifesto.csv).

Uso:  uv run scripts/avaliar_qualidade.py [base ...]      (sem argumentos: todas as bases do manifesto)

Para cada imagem mede nitidez (variância do laplaciano), brilho médio, fração de
pixels estourados/escuros e um hash perceptual para achar quase-duplicatas.
Gera em data/processed/qualidade/:
  qualidade_por_imagem.csv   medidas de cada imagem
  qualidade_por_fonte.csv    resumo por fonte (resolução, nitidez, duplicatas, confirmação do rótulo…)
  amostras_<fonte>.jpg       mosaico com 24 imagens aleatórias da fonte (melanomas na linha de cima)
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

RAIZ = Path(__file__).resolve().parent.parent
PROC = RAIZ / "data" / "processed"
SAIDA = PROC / "qualidade"


def medir(arquivo):
    try:
        im = Image.open(PROC / arquivo).convert("RGB")
        cinza = im.convert("L")
        lap = np.asarray(cinza.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1)), dtype=np.float32)
        g = np.asarray(cinza, dtype=np.float32)
        rgb = np.asarray(im, dtype=np.float32)
        # dHash 8x8: compara pixels vizinhos de uma miniatura 9x8 em tons de cinza
        mini = np.asarray(cinza.resize((9, 8), Image.LANCZOS), dtype=np.int16)
        bits = (mini[:, 1:] > mini[:, :-1]).flatten()
        dhash = "".join("1" if b else "0" for b in bits)
        return {
            "arquivo": arquivo,
            "nitidez": float(lap.var()),
            "brilho": float(g.mean()),
            "frac_estourada": float((g > 245).mean()),
            "frac_escura": float((g < 15).mean()),
            "saturacao": float((rgb.max(axis=2) - rgb.min(axis=2)).mean()),
            "dhash": dhash,
            "erro": "",
        }
    except Exception as erro:
        return {"arquivo": arquivo, "erro": repr(erro)}


def mosaico(df, fonte, lado=160, colunas=8):
    """Mosaico: 1ª linha melanomas, 2ª e 3ª linhas o resto (sorteio fixo)."""
    mel = df[df["classe"] == "MEL"].sample(min(colunas, (df["classe"] == "MEL").sum()), random_state=0)
    resto = df[df["classe"] != "MEL"].sample(min(2 * colunas, (df["classe"] != "MEL").sum()), random_state=0)
    escolhidas = pd.concat([mel, resto])
    linhas = (len(escolhidas) + colunas - 1) // colunas
    tela = Image.new("RGB", (colunas * lado, linhas * lado), "black")
    desenho = ImageDraw.Draw(tela)
    for i, (_, linha) in enumerate(escolhidas.iterrows()):
        im = Image.open(PROC / linha["arquivo"]).convert("RGB")
        im.thumbnail((lado, lado))
        x, y = (i % colunas) * lado, (i // colunas) * lado
        tela.paste(im, (x, y))
        desenho.rectangle([x, y, x + 62, y + 12], fill="black")
        desenho.text((x + 2, y), str(linha["classe"])[:10], fill="yellow")
    tela.save(SAIDA / f"amostras_{fonte}.jpg", quality=85)


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    manifesto = pd.read_csv(PROC / "manifesto.csv", low_memory=False)
    bases = sys.argv[1:] or sorted(manifesto["base"].unique())
    df = manifesto[manifesto["base"].isin(bases) & manifesto["arquivo"].notna()].copy()
    df["fonte"] = df["fonte"].fillna(df["base"])

    with ProcessPoolExecutor(max_workers=6) as executor:
        medidas = pd.DataFrame(list(executor.map(medir, df["arquivo"], chunksize=64)))
    df = df.merge(medidas, on="arquivo")
    df.to_csv(SAIDA / "qualidade_por_imagem.csv", index=False)

    # Quase-duplicatas: mesmo dHash dentro da mesma fonte
    df["dup_hash"] = df.groupby(["fonte", "dhash"])["arquivo"].transform("size") > 1

    linhas = []
    for fonte, g in df.groupby("fonte"):
        confirm = g["confirmacao"].fillna("não informado")
        linhas.append({
            "fonte": fonte,
            "imagens": len(g),
            "melanomas": int((g["classe"] == "MEL").sum()),
            "classes": g["classe"].nunique(),
            "pacientes": g["paciente_id"].nunique() if g["paciente_id"].notna().any() else "sem id",
            "lesoes": g["lesao_id"].nunique() if g["lesao_id"].notna().any() else "sem id",
            "histopatologia_%": round(100 * (confirm == "histopathology").mean()),
            "menor_lado_mediano_px": int(np.minimum(g["largura"], g["altura"]).median()),
            "menor_lado_min_px": int(np.minimum(g["largura"], g["altura"]).min()),
            "nitidez_mediana": round(g["nitidez"].median()),
            "nitidez_p10": round(g["nitidez"].quantile(0.1)),
            "brilho_medio": round(g["brilho"].mean()),
            "estouradas_%": round(100 * (g["frac_estourada"] > 0.05).mean(), 1),
            "escuras_%": round(100 * (g["frac_escura"] > 0.2).mean(), 1),
            "quase_duplicatas_%": round(100 * g["dup_hash"].mean(), 1),
            "falha_leitura": int((g["erro"] != "").sum()),
        })
        mosaico(g, fonte.replace(" ", "_").replace("/", "_"))
    resumo = pd.DataFrame(linhas)
    resumo.to_csv(SAIDA / "qualidade_por_fonte.csv", index=False)
    pd.set_option("display.width", 300)
    print(resumo.to_string(index=False))


if __name__ == "__main__":
    main()

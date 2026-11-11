# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers", "open_clip_torch", "pandas", "pillow", "numpy", "scikit-learn"]
# ///
"""Iteração 1: embeddings de modelos de base congelados para todas as imagens do especialista.

Uso:  uv run triagem_alergias/scripts/extrair_embeddings.py dinov2_base|dermlip_panderm|siglip_base [--tta]

Saída: triagem_alergias/experimentos/embeddings/<modelo>.npy  (N x D, na ordem de dados_scin.tabela_especialista)
       e <modelo>_ids.csv com os image_id, para conferência.
Com --tta, a média do embedding da imagem e da imagem espelhada.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from dados_scin import PASTA, tabela_especialista  # noqa: E402

OUT = PASTA / "experimentos" / "embeddings"
MODELOS = {
    # nome: (como carregar, resolução)
    "dinov2_base": ("hf", "facebook/dinov2-base", 224),
    "dinov2_small": ("hf", "facebook/dinov2-small", 224),
    "siglip_base": ("hf_siglip", "google/siglip-base-patch16-224", 224),
    "dermlip_panderm": ("open_clip", "hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256", 256),  # restrito (401)
    "monet": ("hf_clip", "suinleelab/monet", 224),  # CLIP ViT-L/14 de dermatologia (Kim et al., Nat Med 2024), aberto
}


def carregar(nome):
    tipo, ref, res = MODELOS[nome]
    if tipo == "hf":
        from transformers import AutoImageProcessor, AutoModel
        proc, model = AutoImageProcessor.from_pretrained(ref), AutoModel.from_pretrained(ref).eval()

        def embed(imgs):
            x = proc(images=imgs, return_tensors="pt")
            out = model(**x)
            h = out.last_hidden_state
            return torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)  # CLS + média dos patches (padrão do DINOv2 para linear probe)
    elif tipo == "hf_siglip":
        from transformers import SiglipImageProcessor, SiglipVisionModel
        proc, model = SiglipImageProcessor.from_pretrained(ref), SiglipVisionModel.from_pretrained(ref).eval()

        def embed(imgs):
            x = proc(images=imgs, return_tensors="pt")
            return model(**x).pooler_output
    elif tipo == "hf_clip":
        from transformers import CLIPImageProcessor, CLIPModel
        proc, model = CLIPImageProcessor.from_pretrained(ref), CLIPModel.from_pretrained(ref).eval()

        def embed(imgs):
            x = proc(images=imgs, return_tensors="pt")
            v = model.vision_model(pixel_values=x["pixel_values"])
            return torch.cat([model.visual_projection(v.pooler_output), v.pooler_output], 1)  # projeção CLIP + CLS
    else:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(ref)
        model.eval()

        def embed(imgs):
            x = torch.stack([preprocess(i) for i in imgs])
            return model.encode_image(x)
    return embed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modelo", choices=list(MODELOS))
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--lote", type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(12)
    df, *_ = tabela_especialista()
    OUT.mkdir(parents=True, exist_ok=True)
    embed = carregar(args.modelo)
    feats, t0 = [], time.time()
    with torch.inference_mode():
        for i in range(0, len(df), args.lote):
            imgs = [Image.open(p).convert("RGB") for p in df["arquivo"].iloc[i:i + args.lote]]
            e = embed(imgs)
            if args.tta:
                e = (e + embed([ImageOps.mirror(im) for im in imgs])) / 2
            feats.append(e.float().cpu().numpy())
            if (i // args.lote) % 20 == 0:
                print(f"{i + len(imgs)}/{len(df)}  {(i + len(imgs)) / (time.time() - t0):.1f} img/s", flush=True)
    X = np.concatenate(feats)
    sufixo = "_tta" if args.tta else ""
    np.save(OUT / f"{args.modelo}{sufixo}.npy", X)
    df[["image_id", "case_id", "split", "classe"]].to_csv(OUT / f"{args.modelo}{sufixo}_ids.csv", index=False)
    print("salvo", X.shape, f"em {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

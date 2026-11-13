"""Carregamento das imagens, augmentation e metadados (PyTorch).

As imagens vêm do cache (menor lado 448 px, cores originais). Shades of Gray, recorte e
augmentation são aplicados na hora, porque são parâmetros da busca.
"""
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.io import ImageReadMode, decode_jpeg, read_file
from torchvision.transforms import v2 as T

from .config import CACHE, CLASSES, IDX, SEED, SITES, SPLITS


def ler_splits() -> pd.DataFrame:
    df = pd.read_csv(SPLITS)
    df["y"] = df["dx"].map(IDX)
    df["caminho"] = [str(CACHE / f"{i}.jpg") for i in df["isic_id"]]
    return df


def meta_features(df: pd.DataFrame) -> np.ndarray:
    """Idade, sexo e região do corpo (14 valores), igual ao pipeline.py original."""
    age, sex, site = df["idade"], df["sexo"], df["local"]
    cols = [age.fillna(0) / 100, age.isna(), sex == "female", sex == "male", sex.isna()]
    cols += [site == s for s in SITES] + [site.isna()]
    return np.stack([np.asarray(c, dtype=np.float32) for c in cols], axis=1)


def shades_of_gray(x: torch.Tensor, power: int = 6) -> torch.Tensor:
    """Constância de cor Shades of Gray (Finlayson e Trezzi, 2004) em uint8 CxHxW.

    A iluminação (norma-p de cada canal) é estimada com 1 a cada 4 pixels em cada direção:
    o resultado é praticamente o mesmo e custa 1/16 (era o passo mais caro do carregamento).
    """
    amostra = x[:, ::4, ::4].float() / 255.0
    illum = amostra.pow(power).mean(dim=(1, 2)).pow(1 / power)
    illum = illum / illum.norm()
    return (x.float() / (illum[:, None, None] * math.sqrt(3))).clamp_(0, 255).to(torch.uint8)


class Vinheta(torch.nn.Module):
    """Imita a borda circular preta do dermatoscópio (vem do pipeline.py), para a rede não usar a
    borda como pista da origem da imagem."""
    def __init__(self, p=0.25):
        super().__init__()
        self.p = p

    def forward(self, img):
        if torch.rand(()) >= self.p:
            return img
        h, w = img.shape[-2:]
        r = float(torch.empty(()).uniform_(0.75, 1.15))
        yy = (torch.arange(h) + 0.5) / h * 2 - 1
        xx = (torch.arange(w) + 0.5) / w * 2 - 1
        dist = torch.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)
        mask = torch.sigmoid((r - dist) * 40.0)
        return (img.float() * mask).to(img.dtype)


class Rot90(torch.nn.Module):
    def forward(self, img):
        return torch.rot90(img, int(torch.randint(0, 4, ())), dims=(-2, -1))


class RecorteCentral(torch.nn.Module):
    """Quadrado central do lado menor, depois redimensiona (avaliação, como no pipeline.py)."""
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, img):
        h, w = img.shape[-2:]
        s = min(h, w)
        img = T.functional.center_crop(img, [s, s])
        return T.functional.resize(img, [self.size, self.size], antialias=True)


def transformacoes(size: int, treino: bool, nivel: str = "leve", mean=None, std=None):
    """nivel: leve (= augmentation do pipeline.py original), media ou forte."""
    mean = mean or (0.485, 0.456, 0.406)
    std = std or (0.229, 0.224, 0.225)
    if not treino:
        aug = [RecorteCentral(size)]
    else:
        escala = {"leve": (0.5, 1.0), "media": (0.35, 1.0), "forte": (0.25, 1.0)}[nivel]
        cor = {"leve": (0.1, 0.2, 0.2, 0.02), "media": (0.2, 0.3, 0.3, 0.04), "forte": (0.3, 0.4, 0.4, 0.06)}[nivel]
        aug = [T.RandomResizedCrop(size, scale=escala, ratio=(3 / 4, 4 / 3), antialias=True),
               T.RandomHorizontalFlip(), T.RandomVerticalFlip(), Rot90(),
               T.ColorJitter(*cor)]
        if nivel in ("media", "forte"):
            aug += [T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5)]
        if nivel == "forte":
            aug += [T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.2)]
        aug += [Vinheta(0.25)]
    fim = [T.ToDtype(torch.float32, scale=True), T.Normalize(mean, std)]
    if treino and nivel == "forte":
        fim += [T.RandomErasing(p=0.25, scale=(0.02, 0.1))]
    return T.Compose(aug + fim)


class DermatoDS(Dataset):
    def __init__(self, df: pd.DataFrame, tf, sog: bool):
        self.caminhos = df["caminho"].tolist()
        self.y = df["y"].fillna(-1).astype(int).to_numpy()
        self.meta = torch.from_numpy(meta_features(df))
        self.tf, self.sog = tf, sog

    def __len__(self):
        return len(self.caminhos)

    def __getitem__(self, i):
        img = decode_jpeg(read_file(self.caminhos[i]), mode=ImageReadMode.RGB)
        if self.sog:
            img = shades_of_gray(img)
        return self.tf(img), self.meta[i], self.y[i], i


def limitar_nevos(df: pd.DataFrame, limite, seed=SEED) -> pd.DataFrame:
    if not limite:
        return df
    nv = df[df["dx"] == "nv"]
    if len(nv) <= limite:
        return df
    return pd.concat([df[df["dx"] != "nv"], nv.sample(int(limite), random_state=seed)]).reset_index(drop=True)


def loader(df, size, treino, batch, nivel="leve", sog=True, balanceado=False, mean=None, std=None,
           workers=10, seed=SEED):
    ds = DermatoDS(df, transformacoes(size, treino, nivel, mean, std), sog)
    sampler = None
    if treino and balanceado:
        cont = df["y"].value_counts().reindex(range(len(CLASSES))).fillna(1).to_numpy()
        w = 1.0 / cont[df["y"].to_numpy()]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(df), replacement=True,
                                        generator=torch.Generator().manual_seed(seed))
    return DataLoader(ds, batch_size=batch, shuffle=treino and sampler is None, sampler=sampler,
                      num_workers=workers, pin_memory=True, drop_last=treino,
                      persistent_workers=workers > 0, prefetch_factor=2 if workers else None,
                      generator=torch.Generator().manual_seed(seed) if treino else None)

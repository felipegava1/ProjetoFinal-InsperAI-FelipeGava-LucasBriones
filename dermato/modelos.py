"""Modelos: uma rede (E0, E1, E2) e a fusão de ramos (E3, E4).

Todos recebem (imagem, metadados[, vetor PanDerm]) e devolvem os logits das 8 classes.
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import N_META

BACKBONES = {
    "effv2_b0": "tf_efficientnetv2_b0.in1k",      # E0: a mesma rede do modelo Keras
    "convnext_tiny": "convnext_tiny.fb_in22k",      # E1
    "maxvit_tiny_224": "maxvit_tiny_tf_224.in1k",   # E2 a 224 px
    "maxvit_tiny_384": "maxvit_tiny_tf_384.in1k",   # E2 a 384 px
    "swin_tiny": "swin_tiny_patch4_window7_224.ms_in22k",  # ramo Transformer da E3
}
# lote máximo por passo (antes da acumulação de gradiente), em 224 px, numa GPU de 12 GB
# (MaxViT a 64 por passo estourou os 12 GB com BatchNorm treinável: o Windows passou a usar a RAM
#  do sistema e a época foi de ~2 para 25 min. Lotes maiores viram acumulação de gradiente.)
MICRO_224 = {"effv2_b0": 128, "convnext_tiny": 96, "maxvit_tiny_224": 32, "maxvit_tiny_384": 48, "swin_tiny": 96}
MICRO_FUSAO_224 = 48


def criar_backbone(nome, drop_path=0.0, pretrained=True):
    kw = {"drop_path_rate": drop_path} if drop_path else {}
    return timm.create_model(BACKBONES[nome], pretrained=pretrained, num_classes=0, **kw)


def config_dados(backbone: nn.Module):
    cfg = timm.data.resolve_data_config({}, model=backbone)
    return tuple(cfg["mean"]), tuple(cfg["std"])


class RamoMeta(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_META, dim), nn.SiLU(), nn.Linear(dim, dim), nn.SiLU())
        self.dim = dim

    def forward(self, m):
        return self.net(m)


class Rede(nn.Module):
    """Uma rede + (opcional) metadados, concatenados antes da camada final."""
    def __init__(self, backbone="effv2_b0", usar_meta=True, dropout=0.3, drop_path=0.0, n_classes=8,
                 pretrained=True):
        super().__init__()
        self.backbone = criar_backbone(backbone, drop_path, pretrained)
        self.meta = RamoMeta() if usar_meta else None
        dim = self.backbone.num_features + (self.meta.dim if usar_meta else 0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n_classes))

    def features(self, x, m, emb=None):
        f = self.backbone(x)
        if self.meta is not None:
            f = torch.cat([f, self.meta(m)], 1)
        return f

    def forward(self, x, m, emb=None):
        return self.head(self.features(x, m))


class Fusao(nn.Module):
    """E3: ConvNeXt (detalhe local) + Swin (relações globais) em paralelo, mais metadados.
    E4: + vetor congelado do PanDerm como terceiro ramo (emb_dim > 0).

    juncao:
      concat    concatena os vetores (768 + 768 + 64 [+ PanDerm])
      porta     projeta cada ramo para d e pondera com uma porta aprendida (softmax por imagem)
      atencao   cada ramo vira um token de dimensão d; atenção entre os tokens; média
    """
    def __init__(self, juncao="concat", hidden=512, dropout=0.3, drop_path=0.0, usar_meta=True,
                 emb_dim=0, emb_dropout=0.0, emb_norm=False, n_classes=8, pretrained=True, d=512):
        super().__init__()
        self.cnn = criar_backbone("convnext_tiny", drop_path, pretrained)
        self.vit = criar_backbone("swin_tiny", drop_path, pretrained)
        self.meta = RamoMeta() if usar_meta else None
        self.emb_dim, self.emb_norm = emb_dim, emb_norm
        self.emb_drop = nn.Dropout(emb_dropout)
        self.juncao = juncao
        dims = [self.cnn.num_features, self.vit.num_features] + ([self.meta.dim] if usar_meta else []) \
            + ([emb_dim] if emb_dim else [])
        if juncao == "concat":
            dim_out = sum(dims)
        else:
            self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(k), nn.Linear(k, d)) for k in dims])
            if juncao == "porta":
                self.porta = nn.Linear(sum(dims), len(dims))
            else:
                self.attn = nn.MultiheadAttention(d, num_heads=8, dropout=0.1, batch_first=True)
                self.norm = nn.LayerNorm(d)
            dim_out = d
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim_out, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def ramos(self):
        return {"cnn": self.cnn, "vit": self.vit}

    def features(self, x, m, emb=None):
        x_vit = x if x.shape[-1] == 224 else F.interpolate(x, size=(224, 224), mode="bilinear",
                                                           antialias=True, align_corners=False)
        vs = [self.cnn(x), self.vit(x_vit)]
        if self.meta is not None:
            vs.append(self.meta(m))
        if self.emb_dim:
            e = F.normalize(emb, dim=1) if self.emb_norm else emb
            vs.append(self.emb_drop(e))
        if self.juncao == "concat":
            return torch.cat(vs, 1)
        toks = torch.stack([p(v) for p, v in zip(self.proj, vs)], 1)   # B x R x d
        if self.juncao == "porta":
            w = torch.softmax(self.porta(torch.cat(vs, 1)), 1)          # B x R
            return (toks * w[..., None]).sum(1)
        a, _ = self.attn(toks, toks, toks, need_weights=False)
        return self.norm(toks + a).mean(1)

    def forward(self, x, m, emb=None):
        return self.head(self.features(x, m, emb))


def congelar_bn(model: nn.Module):
    """BatchNorm em modo avaliação e sem gradiente (como no modelo Keras original)."""
    for mod in model.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.eval()
            for p in mod.parameters():
                p.requires_grad_(False)

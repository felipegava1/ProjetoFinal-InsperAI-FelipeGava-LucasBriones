"""Pré-processamento e carregamento das imagens, igual no notebook local e no do Kaggle.

1. build_cache: aplica a constância de cor Shades of Gray e reduz cada imagem para lado menor
   CACHE_SIZE, uma única vez. Uniformiza a cor entre HAM, BCN e MSK e deixa o treino bem mais
   rápido do que decodificar as originais de até 1024x1024 a cada época.
2. make_ds: monta o tf.data. No treino aplica data augmentation; na avaliação recorta o quadrado
   central. As imagens saem em float32 de 0 a 255 (a normalização fica no modelo, porque
   depende da rede usada).
"""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight

from prepare_data import CLASSES, HAM_CLASSES, SEED

CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}
HAM_IDX = {c: i for i, c in enumerate(HAM_CLASSES)}

CACHE_SIZE = 512
SITES = ["anterior torso", "head/neck", "lateral torso", "lower extremity",
         "oral/genital", "palms/soles", "posterior torso", "upper extremity"]
# idade + faltante, sexo (f, m, faltante), região (8 + faltante)
N_META = 2 + 3 + len(SITES) + 1


def shades_of_gray(img: np.ndarray, power: int = 6) -> np.ndarray:
    """Constância de cor Shades of Gray (Finlayson e Trezzi, 2004).

    Estima a cor da iluminação pela norma-p de cada canal e a neutraliza, removendo a
    dominante de cor de cada dermatoscópio/câmera. Usada pelos vencedores do ISIC 2019.
    """
    x = img.astype(np.float32)
    illum = np.power(np.mean(np.power(x, power), axis=(0, 1)), 1 / power)
    illum = illum / np.linalg.norm(illum)
    return np.clip(x / (illum * np.sqrt(3)), 0, 255).astype(np.uint8)


def preprocess_image(src: str, dst: str, size: int = CACHE_SIZE) -> None:
    img = Image.open(src).convert("RGB")
    scale = size / min(img.size)
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.BICUBIC)
    tmp = dst + ".tmp"  # grava e renomeia, para não deixar arquivo pela metade se interromper
    Image.fromarray(shades_of_gray(np.asarray(img))).save(tmp, format="JPEG", quality=95)
    os.replace(tmp, dst)


def build_cache(df: pd.DataFrame, cache_dir, size: int = CACHE_SIZE, workers=None) -> pd.DataFrame:
    """Pré-processa as imagens que ainda não estão no cache e aponta image_path para ele."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = [str(cache_dir / f"{f}.jpg") for f in df["file_id"]]
    todo = [(s, d) for s, d in zip(df["image_path"], dst) if not os.path.exists(d)]
    if todo:
        print(f"Pré-processando {len(todo)} imagens em {cache_dir}...")
        with ThreadPoolExecutor(workers or os.cpu_count()) as ex:
            list(ex.map(lambda a: preprocess_image(*a, size), todo))
    df = df.copy()
    df["image_path"] = dst
    return df


def meta_features(df: pd.DataFrame) -> np.ndarray:
    """Idade, sexo e região do corpo como vetor numérico, com indicadores de faltante."""
    age, sex, site = df["age_approx"], df["sex"], df["site"]
    cols = [age.fillna(0) / 100, age.isna(),
            sex == "female", sex == "male", sex.isna()]
    cols += [site == s for s in SITES] + [site.isna()]
    return np.stack([np.asarray(c, dtype=np.float32) for c in cols], axis=1)


def decode(path):
    return tf.cast(tf.io.decode_jpeg(tf.io.read_file(path), channels=3), tf.float32)


def center_crop(img, size):
    side = tf.reduce_min(tf.shape(img)[:2])
    img = tf.image.resize_with_crop_or_pad(img, side, side)
    return tf.image.resize(img, (size, size))


def random_vignette(img, size):
    """Imita a borda circular preta do dermatoscópio, comum nas imagens do BCN.

    Aplicada em imagens de todas as origens, impede que o modelo use a borda como pista da
    origem (e, por tabela, da classe, já que o BCN tem bem mais melanoma e carcinoma).
    """
    radius = tf.random.uniform([], 0.75, 1.15)
    coords = (tf.range(size, dtype=tf.float32) + 0.5) / size * 2 - 1
    dist = tf.sqrt(coords[None, :] ** 2 + coords[:, None] ** 2)
    mask = tf.sigmoid((radius - dist) * 40.0)[..., None]
    return img * mask


def augment(img, size):
    begin, crop, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(img), bounding_boxes=tf.zeros([1, 0, 4]), min_object_covered=0.0,
        # área mínima de 50%: recortes menores às vezes deixavam a lesão de fora
        aspect_ratio_range=(3 / 4, 4 / 3), area_range=(0.5, 1.0), max_attempts=10,
        use_image_if_no_bounding_boxes=True)
    img = tf.image.resize(tf.slice(img, begin, crop), (size, size))
    # lesões não têm orientação: flips e rotações de 90° são todas válidas
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, tf.random.uniform([], 0, 4, dtype=tf.int32))
    x = img / 255.0
    x = tf.image.random_brightness(x, 0.1)
    x = tf.image.random_contrast(x, 0.8, 1.2)
    x = tf.image.random_saturation(x, 0.8, 1.2)
    x = tf.image.random_hue(x, 0.02)
    img = tf.clip_by_value(x, 0.0, 1.0) * 255.0
    return tf.cond(tf.random.uniform([]) < 0.25, lambda: random_vignette(img, size), lambda: img)


def _talvez(p, fn, img):
    return tf.cond(tf.random.uniform([]) < p, lambda: fn(img), lambda: img)


def _perda_resolucao(img, size):
    """Foto de longe / sensor fraco: reduz para 25–70% e amplia de volta."""
    novo = tf.cast(tf.round(tf.cast(size, tf.float32) * tf.random.uniform([], 0.25, 0.7)), tf.int32)
    pequena = tf.image.resize(img, (novo, novo), method="area")
    return tf.image.resize(pequena, (size, size), method="bilinear")


def _desfoque(img):
    """Desfoque gaussiano separável, sigma 0,5–2,5 px."""
    sigma = tf.random.uniform([], 0.5, 2.5)
    x = tf.range(-4, 5, dtype=tf.float32)
    k = tf.exp(-0.5 * (x / sigma) ** 2)
    k = k / tf.reduce_sum(k)
    kh = tf.tile(tf.reshape(k, [1, 9, 1, 1]), [1, 1, 3, 1])
    kv = tf.tile(tf.reshape(k, [9, 1, 1, 1]), [1, 1, 3, 1])
    x4 = tf.nn.depthwise_conv2d(img[None], kh, [1, 1, 1, 1], "SAME")
    return tf.nn.depthwise_conv2d(x4, kv, [1, 1, 1, 1], "SAME")[0]


def _ruido(img):
    """Ruído de sensor gaussiano, sigma 3–14 em 0–255."""
    return img + tf.random.normal(tf.shape(img)) * tf.random.uniform([], 3.0, 14.0)


def _jpeg(img):
    """Compressão JPEG de qualidade 25–80 (apps e mensageiros recomprimem)."""
    q = tf.random.uniform([], 25, 80, dtype=tf.int32)
    u8 = tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8)
    return tf.cast(tf.image.adjust_jpeg_quality(u8, q), tf.float32)


def _balanco_de_branco(img):
    """Erro de balanço de branco: ganho por canal 0,8–1,2 (luz amarela/azulada)."""
    return img * tf.random.uniform([3], 0.8, 1.2)


def _cor_forte(img):
    x = tf.image.random_brightness(img / 255.0, 0.25)
    x = tf.image.random_contrast(x, 0.6, 1.4)
    x = tf.image.random_saturation(x, 0.6, 1.4)
    x = tf.image.random_hue(x, 0.05)
    return tf.clip_by_value(x, 0, 1) * 255.0


def _sombra(img, size):
    """Gradiente linear de iluminação (sombra da mão / luz lateral)."""
    ang = tf.random.uniform([], 0, 2 * np.pi)
    forca = tf.random.uniform([], 0.2, 0.45)
    coords = (tf.range(size, dtype=tf.float32) / size) - 0.5
    proj = tf.cos(ang) * coords[None, :] + tf.sin(ang) * coords[:, None]
    return img * (1.0 - forca * (proj + 0.5))[..., None]


def _afastar(img, size):
    """Lesão menor no quadro: encolhe para 50–90% e completa com a cor da pele, mais ruído."""
    novo = tf.cast(tf.round(tf.cast(size, tf.float32) * tf.random.uniform([], 0.5, 0.9)), tf.int32)
    pequena = tf.image.resize(img, (novo, novo))
    borda = tf.concat([tf.reshape(img[:8], [-1, 3]), tf.reshape(img[-8:], [-1, 3]),
                       tf.reshape(img[:, :8], [-1, 3]), tf.reshape(img[:, -8:], [-1, 3])], 0)
    # cor da pele = média dos pixels claros da borda (ignora a borda preta do dermatoscópio)
    claros = tf.boolean_mask(borda, tf.reduce_mean(borda, 1) > 70.0)
    pele = tf.cond(tf.shape(claros)[0] > 20, lambda: tf.reduce_mean(claros, 0),
                   lambda: tf.constant([205.0, 160.0, 140.0]))
    fundo = tf.ones([size, size, 3]) * pele + tf.random.normal([size, size, 3]) * 6.0
    dx = tf.random.uniform([], 0, size - novo + 1, dtype=tf.int32)
    dy = tf.random.uniform([], 0, size - novo + 1, dtype=tf.int32)
    margens = [[dy, size - novo - dy], [dx, size - novo - dx], [0, 0]]
    mask = tf.pad(tf.ones([novo, novo, 1]), margens)
    return tf.pad(pequena, margens) * mask + fundo * (1 - mask)


def _recorte_interno(img, size):
    """Tira a borda circular do dermatoscópio: recorte central de 70–90% e volta ao tamanho."""
    lado = tf.cast(tf.random.uniform([], 0.7, 0.9) * size, tf.int32)
    canto = (size - lado) // 2
    return tf.image.resize(tf.image.crop_to_bounding_box(img, canto, canto, lado, lado), (size, size))


def simular_celular(img, size=224):
    """Faz uma imagem de dermatoscopia (ou câmera) parecer foto de celular.

    É a ideia central do modelo do projeto: aprender a reconhecer lesões em foto de celular sem
    usar nenhuma foto de celular no treino. Cada efeito é sorteado com a sua probabilidade. Vem
    depois de `augment` e espera a imagem já em size x size, float de 0 a 255.
    """
    img = _talvez(0.5, lambda x: _recorte_interno(x, size), img)
    img = _talvez(0.4, lambda x: _afastar(x, size), img)
    img = _talvez(0.6, _cor_forte, img)
    img = _talvez(0.5, _balanco_de_branco, img)
    img = _talvez(0.4, lambda x: _sombra(x, size), img)
    img = _talvez(0.5, lambda x: _perda_resolucao(x, size), img)
    img = _talvez(0.4, _desfoque, img)
    img = _talvez(0.5, _ruido, img)
    img = _talvez(0.5, _jpeg, img)
    return tf.clip_by_value(img, 0.0, 255.0)


def make_ds(df: pd.DataFrame, img_size=224, batch_size=32, training=False, labels="dx",
            use_meta=False):
    """tf.data a partir do DataFrame.

    labels: "dx" (índices de CLASSES), "dx_ham" (índices de HAM_CLASSES) ou None (só entradas).
    Com use_meta, as entradas são {"image", "meta"}; sem, só a imagem.
    """
    x = {"image": df["image_path"].values}
    if use_meta:
        x["meta"] = meta_features(df)
    if labels is not None:
        idx = CLASS_IDX if labels == "dx" else HAM_IDX
        ds = tf.data.Dataset.from_tensor_slices((x, df[labels].map(idx).values))
    else:
        ds = tf.data.Dataset.from_tensor_slices((x,))
    if training:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)

    def load(inp, *y):
        img = decode(inp["image"])
        img = augment(img, img_size) if training else center_crop(img, img_size)
        out = {"image": img, "meta": inp["meta"]} if use_meta else img
        return (out, *y) if y else out

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size, drop_remainder=training).prefetch(tf.data.AUTOTUNE)


def make_datasets(df: pd.DataFrame, img_size=224, batch_size=32, use_meta=False) -> dict:
    """train (com augmentation) e val com rótulos em CLASSES, test com rótulos em HAM_CLASSES,
    unlabeled sem rótulo."""
    part = {s: df[df["split"] == s] for s in ["train", "val", "test", "unlabeled"]}
    kw = dict(img_size=img_size, batch_size=batch_size, use_meta=use_meta)
    return {
        "train": make_ds(part["train"], training=True, **kw),
        "val": make_ds(part["val"], **kw),
        "test": make_ds(part["test"], labels="dx_ham", **kw),
        "unlabeled": make_ds(part["unlabeled"], labels=None, **kw),
    }


def class_weights(dx: pd.Series) -> dict:
    """Pesos "balanced" por índice de classe, para o class_weight do model.fit."""
    w = compute_class_weight("balanced", classes=np.arange(len(CLASSES)), y=dx.map(CLASS_IDX))
    return dict(enumerate(w.tolist()))


def locate_images(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Aponta image_path para onde cada arquivo está dentro de root, procurando pelo nome.

    Serve para o Kaggle, onde as imagens vêm de datasets diferentes com pastas diferentes.
    """
    found = {}
    # os.walk com followlinks: as pastas de /kaggle/input podem ser links simbólicos
    for folder, _, files in os.walk(root, followlinks=True):
        for name in files:
            if name.endswith(".jpg"):
                found.setdefault(name, os.path.join(folder, name))
    df = df.copy()
    df["image_path"] = (df["file_id"] + ".jpg").map(found)
    return df

"""Carregamento das imagens em tf.data, igual no notebook local e no do Kaggle.

As imagens saem redimensionadas, em float32 de 0 a 255; normalização e data augmentation
ficam no modelo, porque dependem da rede usada.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from prepare_data import CLASSES, HAM_CLASSES, SEED

CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}
HAM_IDX = {c: i for i, c in enumerate(HAM_CLASSES)}


def load_image(path, img_size):
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    return tf.image.resize(img, (img_size, img_size))


def make_ds(paths, labels=None, img_size=224, batch_size=32, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices(paths if labels is None else (paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    if labels is None:
        fn = lambda p: load_image(p, img_size)
    else:
        fn = lambda p, y: (load_image(p, img_size), y)
    return ds.map(fn, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_datasets(df: pd.DataFrame, img_size=224, batch_size=32) -> dict:
    """train/val com rótulos em CLASSES, test com rótulos em HAM_CLASSES, unlabeled sem rótulo."""
    part = {s: df[df["split"] == s] for s in ["train", "val", "test", "unlabeled"]}
    kw = dict(img_size=img_size, batch_size=batch_size)
    return {
        "train": make_ds(part["train"]["image_path"].values,
                         part["train"]["dx"].map(CLASS_IDX).values, shuffle=True, **kw),
        "val": make_ds(part["val"]["image_path"].values, part["val"]["dx"].map(CLASS_IDX).values, **kw),
        "test": make_ds(part["test"]["image_path"].values,
                        part["test"]["dx_ham"].map(HAM_IDX).values, **kw),
        "unlabeled": make_ds(part["unlabeled"]["image_path"].values, **kw),
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

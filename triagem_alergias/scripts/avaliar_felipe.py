# /// script
# requires-python = ">=3.12"
# dependencies = ["tensorflow-cpu", "pandas", "pillow", "numpy", "scikit-learn", "scipy"]
# ///
"""Iteração 0: o especialista do Felipe (modelos/triagem/especialista_erupcao.keras) avaliado no mesmo
teste, com as imagens da SCIN deste PC. Serve para confirmar que a comparação é justa.

Saída: triagem_alergias/experimentos/felipe/{probs_val.npy, probs_test.npy, metricas.json}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parents[1]))
from dados_scin import PASTA, tabela_especialista  # noqa: E402
from metricas import metricas_especialista, resumo  # noqa: E402

OUT = PASTA / "experimentos" / "felipe"
IMG = 224


def main():
    import tensorflow as tf
    import keras
    from scipy.special import softmax
    from pipeline import center_crop, decode  # noqa: E402

    tf.config.threading.set_intra_op_parallelism_threads(12)
    df, classes, alvos, meta = tabela_especialista()
    lim = json.loads((PASTA / "modelos" / "triagem" / "limiares.json").read_text(encoding="utf-8"))["especialista_erupcao"]
    assert lim["classes"] == classes, "ordem de classes diferente da do notebook"
    model = keras.models.load_model(PASTA / "modelos" / "triagem" / "especialista_erupcao.keras", compile=False)
    OUT.mkdir(parents=True, exist_ok=True)
    y = df["classe"].map({c: i for i, c in enumerate(classes)}).values
    res = {}
    for split in ("val", "test"):
        m = (df["split"] == split).values
        ds = tf.data.Dataset.from_tensor_slices(df["arquivo"][m].values).map(
            lambda p: center_crop(decode(p), IMG), num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(2)
        logits = model.predict(ds, verbose=0)
        probs = softmax(logits / lim["temperatura"], axis=1)
        np.save(OUT / f"probs_{split}.npy", probs)
        res[split] = metricas_especialista(probs, y[m], df["case_id"][m].values, classes)
        print(split, resumo(res[split]), flush=True)
    (OUT / "metricas.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("referência do Felipe (teste): top1 0.283 | top3 0.692 | bal_acc 0.170 | f1 0.182 | auroc 0.806 | por caso 0.334")


if __name__ == "__main__":
    main()

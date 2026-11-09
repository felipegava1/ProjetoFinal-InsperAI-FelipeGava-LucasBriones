# /// script
# requires-python = ">=3.12"
# dependencies = ["tensorflow-cpu", "pandas", "pillow", "numpy", "scikit-learn"]
# ///
"""Treina SEM fotos de celular e testa NAS fotos de celular, usando augmentation que simula
celular (resolução, desfoque, ruído, JPEG, balanço de branco, sombra, afastamento).

Uso:  uv run scripts/treinar_simulando_celular.py --conjunto dermatoscopia|dermatoscopia+fotos
                                                  [--sem-simulacao] [--epocas 8] [--lr 1e-4] [--rapido]

  dermatoscopia        ISIC 2019 (treino), ISIC 2020 (rotuladas) e MILK10k dermatoscopia
  dermatoscopia+fotos  + fotos clínicas de câmera (MILK10k foto, MED-NODE, MSKCC, anônima)
  --sem-simulacao      só a augmentation original do pipeline.py (controle)

Parte do professor do Felipe (modelos/professor_effv2b0/melhor.keras). Validação, escolha da época
e do tau: só nos splits de validação das mesmas bases (nenhuma foto de celular toca o treino).
Teste: PAD-UFES-20 e HIBA inteiras (2.653 fotos, 117 melanomas), com TTA. Para acompanhar, a AUC
de melanoma no celular é impressa a cada época, mas NÃO é usada para escolher nada.
Nevos do treino são limitados a --max-nevos por época (sorteio fixo), para acelerar e reequilibrar.

Saída em data/modelos/simulando_celular/<conjunto>[_sem_simulacao]/: melhor.keras, historico.csv,
comparacao.csv, relatorio.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
PROC = RAIZ / "data" / "processed"
CACHE = PROC / "cache_sog_256"
OUT = RAIZ / "data" / "modelos" / "simulando_celular"
MODELO_FELIPE = RAIZ / "modelos" / "professor_effv2b0" / "melhor.keras"
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_data import CLASSES  # noqa: E402
from pipeline import CLASS_IDX, augment, build_cache, center_crop, decode, meta_features  # noqa: E402
from testar_modelo import PARA_FELIPE, PARA_SITE, definir_grupos, softmax  # noqa: E402

SEED = 42
IMG_SIZE, BATCH_SIZE, WEIGHT_DECAY, TTA_VIEWS = 224, 32, 0.02, 8
MALIGNAS = ["ak", "bcc", "mel", "scc"]
CELULAR = ["pad_ufes_20", "hiba"]
CONJUNTOS = {
    "dermatoscopia": ["isic2019_train", "isic2020", "milk10k_dermatoscopia"],
    "dermatoscopia+fotos": ["isic2019_train", "isic2020", "milk10k_dermatoscopia", "milk10k_foto", "mednode", "mskcc", "anonima"],
}


def montar_tabela(fontes, max_nevos, rapido):
    m = pd.read_csv(PROC / "manifesto.csv", low_memory=False)
    m["grupo"] = definir_grupos(m)
    m.loc[(m["base"] == "isic2019") & (m["particao_oficial"] == "train"), "grupo"] = "isic2019_train"
    m = m[m["arquivo"].notna() & m["duplicada_de"].isna() & m["classe"].isin(PARA_FELIPE)]
    df = pd.DataFrame({
        "file_id": m["isic_id"].values, "image_path": [str(PROC / a) for a in m["arquivo"]],
        "dx": m["classe"].map(PARA_FELIPE).values, "grupo": m["grupo"].values, "split": m["split"].values,
        "age_approx": m["idade"].values, "sex": m["sexo"].where(m["sexo"].isin(["male", "female"])).values,
        "site": m["local_anatomico"].map(PARA_SITE).values,
    })
    treino = df[df["grupo"].isin(fontes) & (df["split"] == "train")]
    nevos = treino[treino["dx"] == "nv"]
    if len(nevos) > max_nevos:
        treino = pd.concat([treino[treino["dx"] != "nv"], nevos.sample(max_nevos, random_state=SEED)])
    val = df[df["grupo"].isin(fontes) & (df["split"] == "val")]
    teste = df[df["grupo"].isin(CELULAR)]  # inteiras: nenhuma foto de celular entra no treino
    if rapido:
        treino, val, teste = treino.groupby("dx").head(6), val.groupby("dx").head(4), teste.groupby(["grupo", "dx"]).head(3)
    return [d.sample(frac=1, random_state=SEED).reset_index(drop=True) for d in (treino, val, teste)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conjunto", choices=list(CONJUNTOS), default="dermatoscopia")
    ap.add_argument("--sem-simulacao", action="store_true")
    ap.add_argument("--epocas", type=int, default=8)
    ap.add_argument("--paciencia", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-nevos", type=int, default=6000)
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--adabn", metavar="MODELO.keras",
                    help="plano B: em vez de treinar, reestima as BatchNormalization do modelo dado nas fotos de celular (sem rótulo)")
    ap.add_argument("--nome", help="nome da pasta de saída (padrão: derivado do conjunto)")
    args = ap.parse_args()

    import tensorflow as tf
    import keras
    from keras import callbacks, layers, optimizers
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, roc_auc_score

    keras.utils.set_random_seed(SEED)
    tf.config.threading.set_intra_op_parallelism_threads(14)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    nome_var = args.nome or (args.conjunto + ("_sem_simulacao" if args.sem_simulacao else ""))
    out = OUT / (nome_var + ("_rapido" if args.rapido else ""))
    out.mkdir(parents=True, exist_ok=True)
    batch, epochs, tta = (8, 1, 2) if args.rapido else (BATCH_SIZE, args.epocas, TTA_VIEWS)

    treino, val, teste = montar_tabela(CONJUNTOS[args.conjunto], args.max_nevos, args.rapido)
    for nome, d in [("treino", treino), ("validação", val), ("teste celular", teste)]:
        print(f"{nome}: {len(d)} | grupos {d['grupo'].value_counts().to_dict()} | classes {d['dx'].value_counts().to_dict()}", flush=True)
    todas = pd.concat([treino, val, teste]).drop_duplicates("file_id")
    todas = build_cache(todas, CACHE, size=256)
    caminho = dict(zip(todas["file_id"], todas["image_path"]))
    for d in (treino, val, teste):
        d["image_path"] = d["file_id"].map(caminho)
    y_val = val["dx"].map(CLASS_IDX).values
    y_teste = teste["dx"].map(CLASS_IDX).values
    prior = treino["dx"].value_counts(normalize=True).reindex(CLASSES).fillna(1e-3).values
    log_prior = np.log(prior)

    # ------------------------------------------------------------------ simulação de celular
    def talvez(p, fn, img):
        return tf.cond(tf.random.uniform([]) < p, lambda: fn(img), lambda: img)

    def perda_resolucao(img):
        """Foto de longe / sensor fraco: reduz para 25–70 % e amplia de volta."""
        f = tf.random.uniform([], 0.25, 0.7)
        novo = tf.cast(tf.round(tf.cast(IMG_SIZE, tf.float32) * f), tf.int32)
        pequena = tf.image.resize(img, (novo, novo), method="area")
        return tf.image.resize(pequena, (IMG_SIZE, IMG_SIZE), method="bilinear")

    def desfoque(img):
        """Desfoque gaussiano separável, sigma 0,5–2,5 px."""
        sigma = tf.random.uniform([], 0.5, 2.5)
        x = tf.range(-4, 5, dtype=tf.float32)
        k = tf.exp(-0.5 * (x / sigma) ** 2)
        k = k / tf.reduce_sum(k)
        kh = tf.tile(tf.reshape(k, [1, 9, 1, 1]), [1, 1, 3, 1])
        kv = tf.tile(tf.reshape(k, [9, 1, 1, 1]), [1, 1, 3, 1])
        x4 = img[None]
        x4 = tf.nn.depthwise_conv2d(x4, kh, [1, 1, 1, 1], "SAME")
        x4 = tf.nn.depthwise_conv2d(x4, kv, [1, 1, 1, 1], "SAME")
        return x4[0]

    def ruido(img):
        """Ruído de sensor (gaussiano, mais forte em foto escura), sigma 3–14 em 0–255."""
        sigma = tf.random.uniform([], 3.0, 14.0)
        return img + tf.random.normal(tf.shape(img)) * sigma

    def jpeg(img):
        """Compressão JPEG de qualidade 25–80 (apps e mensageiros recomprimem)."""
        q = tf.random.uniform([], 25, 80, dtype=tf.int32)
        u8 = tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8)
        return tf.cast(tf.image.adjust_jpeg_quality(u8, q), tf.float32)

    def balanco_de_branco(img):
        """Erro de balanço de branco: ganho por canal 0,8–1,2 (luz amarela/azulada)."""
        ganho = tf.random.uniform([3], 0.8, 1.2)
        return img * ganho

    def cor_forte(img):
        x = img / 255.0
        x = tf.image.random_brightness(x, 0.25)
        x = tf.image.random_contrast(x, 0.6, 1.4)
        x = tf.image.random_saturation(x, 0.6, 1.4)
        x = tf.image.random_hue(x, 0.05)
        return tf.clip_by_value(x, 0, 1) * 255.0

    def sombra(img):
        """Gradiente linear de iluminação (sombra da mão / luz lateral), de 0,55 a 1,0."""
        ang = tf.random.uniform([], 0, 2 * np.pi)
        forca = tf.random.uniform([], 0.2, 0.45)
        coords = (tf.range(IMG_SIZE, dtype=tf.float32) / IMG_SIZE) - 0.5
        proj = tf.cos(ang) * coords[None, :] + tf.sin(ang) * coords[:, None]
        mask = 1.0 - forca * (proj + 0.5)
        return img * mask[..., None]

    def afastar(img):
        """Lesão menor no quadro: encolhe para 50–90 % e preenche com a cor média da borda
        (pele) mais ruído, como numa foto tirada de mais longe."""
        f = tf.random.uniform([], 0.5, 0.9)
        novo = tf.cast(tf.round(tf.cast(IMG_SIZE, tf.float32) * f), tf.int32)
        pequena = tf.image.resize(img, (novo, novo))
        borda = tf.concat([tf.reshape(img[:8], [-1, 3]), tf.reshape(img[-8:], [-1, 3]),
                           tf.reshape(img[:, :8], [-1, 3]), tf.reshape(img[:, -8:], [-1, 3])], 0)
        # cor da pele = média dos pixels claros da borda (ignora a borda preta do dermatoscópio)
        claros = tf.boolean_mask(borda, tf.reduce_mean(borda, 1) > 70.0)
        pele = tf.cond(tf.shape(claros)[0] > 20, lambda: tf.reduce_mean(claros, 0),
                       lambda: tf.constant([205.0, 160.0, 140.0]))
        fundo = tf.ones([IMG_SIZE, IMG_SIZE, 3]) * pele + tf.random.normal([IMG_SIZE, IMG_SIZE, 3]) * 6.0
        dx = tf.random.uniform([], 0, IMG_SIZE - novo + 1, dtype=tf.int32)
        dy = tf.random.uniform([], 0, IMG_SIZE - novo + 1, dtype=tf.int32)
        pad = tf.pad(pequena, [[dy, IMG_SIZE - novo - dy], [dx, IMG_SIZE - novo - dx], [0, 0]])
        mask = tf.pad(tf.ones([novo, novo, 1]), [[dy, IMG_SIZE - novo - dy], [dx, IMG_SIZE - novo - dx], [0, 0]])
        return pad * mask + fundo * (1 - mask)

    def recorte_interno(img):
        """Tira a borda circular do dermatoscópio: recorte central de 70–90 % e volta a 224."""
        f = tf.random.uniform([], 0.7, 0.9)
        lado = tf.cast(f * IMG_SIZE, tf.int32)
        canto = (IMG_SIZE - lado) // 2
        return tf.image.resize(tf.image.crop_to_bounding_box(img, canto, canto, lado, lado), (IMG_SIZE, IMG_SIZE))

    def simular_celular(img):
        img = talvez(0.5, recorte_interno, img)
        img = talvez(0.4, afastar, img)
        img = talvez(0.6, cor_forte, img)
        img = talvez(0.5, balanco_de_branco, img)
        img = talvez(0.4, sombra, img)
        img = talvez(0.5, perda_resolucao, img)
        img = talvez(0.4, desfoque, img)
        img = talvez(0.5, ruido, img)
        img = talvez(0.5, jpeg, img)
        return tf.clip_by_value(img, 0.0, 255.0)

    def make_ds(df, training):
        x = {"image": df["image_path"].values, "meta": meta_features(df)}
        y = df["dx"].map(CLASS_IDX).values
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if training:
            ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)

        def load(inp, yy):
            img = decode(inp["image"])
            if training:
                img = augment(img, IMG_SIZE)
                if not args.sem_simulacao:
                    img = simular_celular(img)
            else:
                img = center_crop(img, IMG_SIZE)
            return {"image": img, "meta": inp["meta"]}, yy

        ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(batch, drop_remainder=training).prefetch(tf.data.AUTOTUNE)

    # ------------------------------------------------------------------ utilidades
    def tta_view(x, k):
        img = tf.image.rot90(x["image"], k % 4)
        if k >= 4:
            img = tf.image.flip_left_right(img)
        return {**x, "image": img}

    def predict_tta(model, df, views):
        ds = make_ds(df, training=False).map(lambda x, y: x)
        probs = 0
        for k in range(views):
            probs = probs + softmax(model.predict(ds.map(lambda x, k=k: tta_view(x, k)), verbose=0))
        return probs / views

    def ajustar(p, tau):
        return softmax(np.log(p + 1e-12) - tau * log_prior)

    def tune_tau(p_val):
        taus = np.arange(0, 2.01, 0.05)
        return float(taus[int(np.argmax([balanced_accuracy_score(y_val, ajustar(p_val, t).argmax(1)) for t in taus]))])

    val_ds = make_ds(val, training=False)
    teste_ds_1 = make_ds(teste, training=False).map(lambda x, y: x)

    class ValMetrics(callbacks.Callback):
        """Acurácia balanceada da validação (decide o checkpoint) e, só para acompanhar,
        AUC de melanoma no celular."""
        def on_epoch_end(self, epoch, logs=None):
            logits = self.model.predict(val_ds.map(lambda x, y: x), verbose=0)
            logs["val_bal_acc"] = balanced_accuracy_score(y_val, (logits - log_prior).argmax(1))
            p_cel = softmax(self.model.predict(teste_ds_1, verbose=0))
            mel = y_teste == CLASS_IDX["mel"]
            logs["celular_mel_auc_informativo"] = roc_auc_score(mel, p_cel[:, CLASS_IDX["mel"]])
            logs["celular_bal_acc_informativo"] = balanced_accuracy_score(y_teste, (np.log(p_cel + 1e-12) - 0.9 * log_prior).argmax(1))
            print(f"  época {epoch + 1}: val_loss={logs.get('val_loss', float('nan')):.4f} val_bal_acc={logs['val_bal_acc']:.4f} "
                  f"| celular (só informativo): AUC mel={logs['celular_mel_auc_informativo']:.3f} "
                  f"bal_acc={logs['celular_bal_acc_informativo']:.3f}", flush=True)

    # ------------------------------------------------------------------ treino
    ckpt, hist = out / "melhor.keras", out / "historico.csv"
    if args.adabn and not (out / "concluido").exists():
        # Plano B: AdaBN. Sem rótulo e sem gradiente: só as médias/variâncias das BatchNormalization
        # são reestimadas em fotos de celular (todas as outras pesos ficam iguais).
        model = keras.models.load_model(args.adabn, compile=False)
        base = next(l for l in model.layers if "efficientnet" in l.name.lower())
        bns = [l for l in base.layers if isinstance(l, layers.BatchNormalization)]
        for l in bns:
            l.trainable = True
            l.momentum = 0.9
        ds = make_ds(teste, training=False).map(lambda x, y: x)
        for passada in range(3):
            for lote in ds:
                base(lote["image"], training=True)  # atualiza só as estatísticas móveis
        for l in bns:
            l.trainable = False
        model.save(ckpt)
        (out / "concluido").write_text("adabn de " + str(args.adabn))
        del model
        keras.backend.clear_session()
    if not (out / "concluido").exists():
        model = keras.models.load_model(MODELO_FELIPE, compile=False)
        base = next(l for l in model.layers if "efficientnet" in l.name.lower())
        base.trainable = True
        for layer in base.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
        train_ds = make_ds(treino, training=True)
        per_epoch = int(train_ds.cardinality())
        # Mosaico de conferência: 1ª linha originais (recorte central), 2ª e 3ª a mesma imagem após a augmentation
        from PIL import Image
        amostra = treino.head(8)
        orig = np.stack([center_crop(decode(p), IMG_SIZE).numpy() for p in amostra["image_path"]])
        def exemplo(p):
            img = augment(decode(p), IMG_SIZE)
            return (img if args.sem_simulacao else simular_celular(img)).numpy()
        aug = [np.stack([exemplo(p) for p in amostra["image_path"]]) for _ in range(2)]
        tela = np.concatenate([orig] + aug, axis=0).reshape(3, 8, IMG_SIZE, IMG_SIZE, 3).transpose(0, 2, 1, 3, 4)
        Image.fromarray(np.clip(tela.reshape(3 * IMG_SIZE, 8 * IMG_SIZE, 3), 0, 255).astype(np.uint8)).save(out / "augment_exemplos.jpg", quality=85)
        schedule = optimizers.schedules.CosineDecay(args.lr / 20, decay_steps=max(1, per_epoch * (epochs - 1)),
                                                    alpha=0.01, warmup_target=args.lr, warmup_steps=per_epoch)
        opt = optimizers.AdamW(schedule, weight_decay=WEIGHT_DECAY)
        opt.exclude_from_weight_decay(var_names=["bias", "gamma", "beta"])
        model.compile(optimizer=opt, loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        melhor_anterior = pd.read_csv(hist)["val_bal_acc"].max() if hist.exists() and hist.stat().st_size else None
        print(f"Treino [{nome_var}]: {len(treino)} imagens, {per_epoch} passos/época, lr {args.lr}", flush=True)
        model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2, callbacks=[
            callbacks.BackupAndRestore(str(out / "backup")), ValMetrics(),
            callbacks.ModelCheckpoint(str(ckpt), monitor="val_bal_acc", mode="max", save_best_only=True,
                                      initial_value_threshold=melhor_anterior),
            callbacks.EarlyStopping(monitor="val_bal_acc", mode="max", patience=args.paciencia),
            callbacks.CSVLogger(str(hist), append=True)])
        (out / "concluido").write_text("ok")
        del model
        keras.backend.clear_session()

    # ------------------------------------------------------------------ comparação no celular
    def metricas(mask, p, pred, tau, limiar):
        yy, pp, pm = y_teste[mask], pred[mask], p[mask]
        mel, e_mel = yy == CLASS_IDX["mel"], pp == CLASS_IDX["mel"]
        mal = np.isin(yy, [CLASS_IDX[c] for c in MALIGNAS])
        s = pm[:, [CLASS_IDX[c] for c in MALIGNAS]].sum(1)
        return {"imagens": int(mask.sum()), "melanomas": int(mel.sum()), "tau": tau, "limiar": round(limiar, 3),
                "acuracia": accuracy_score(yy, pp), "acuracia_balanceada": balanced_accuracy_score(yy, pp),
                "mel_sensibilidade": float(e_mel[mel].mean()) if mel.any() else np.nan,
                "mel_precisao": float(mel[e_mel].mean()) if e_mel.any() else np.nan,
                "mel_auc": roc_auc_score(mel, pm[:, CLASS_IDX["mel"]]) if 0 < mel.sum() < len(yy) else np.nan,
                "maligno_auc": roc_auc_score(mal, s) if 0 < mal.sum() < len(yy) else np.nan,
                "triagem_sensibilidade": float((s[mal] >= limiar).mean()), "triagem_especificidade": float((s[~mal] < limiar).mean())}

    linhas, relatorio = [], []
    previsoes = teste[["file_id", "grupo", "dx"]].copy()
    for nome, caminho_modelo in [("antes (Felipe)", MODELO_FELIPE), (f"depois ({nome_var})", ckpt)]:
        print(f"Avaliando {nome}", flush=True)
        model = keras.models.load_model(caminho_modelo, compile=False)
        p_val = predict_tta(model, val, 1)
        tau = tune_tau(p_val)
        s_val = p_val[:, [CLASS_IDX[c] for c in MALIGNAS]].sum(1)
        limiar = float(np.quantile(s_val[np.isin(y_val, [CLASS_IDX[c] for c in MALIGNAS])], 0.05))
        p = predict_tta(model, teste, tta)
        pred = ajustar(p, tau).argmax(1)
        for i, c in enumerate(CLASSES):
            previsoes[f"{nome.split()[0]}_p_{c}"] = p[:, i]
        for conj, mask in {"celular (PAD-UFES-20 + HIBA)": np.ones(len(teste), bool),
                           "pad_ufes_20": (teste["grupo"] == "pad_ufes_20").values,
                           "hiba": (teste["grupo"] == "hiba").values}.items():
            linhas.append({"modelo": nome, "conjunto": conj} | metricas(mask, p, pred, tau, limiar))
        presentes = sorted(set(y_teste) | set(pred))
        relatorio.append(f"=== {nome}: celular, tau={tau} ===\n" + classification_report(
            y_teste, pred, labels=presentes, target_names=[CLASSES[i] for i in presentes], digits=3, zero_division=0))
        del model
        keras.backend.clear_session()

    comp = pd.DataFrame(linhas)
    comp.to_csv(out / "comparacao.csv", index=False)
    previsoes.to_csv(out / "previsoes_celular.csv", index=False)
    pd.set_option("display.width", 250)
    texto = comp.round(3).to_string(index=False) + "\n\n" + "\n".join(relatorio)
    (out / "relatorio.txt").write_text(texto, encoding="utf-8")
    print("\n" + texto)
    print("Saídas em", out)


if __name__ == "__main__":
    main()

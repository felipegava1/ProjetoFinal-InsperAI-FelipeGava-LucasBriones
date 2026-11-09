# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "pillow", "scikit-learn"]
# ///
"""Deixa as bases prontas para treino.

Uso:  uv run scripts/preparar_dados.py [base ...]      (sem argumentos: todas)
      Bases: scin isic2018 isic2019 isic2020 isic2024 milk10k isic_clinicas mednode.
      Cada base processada fica em cache; o manifesto final junta todas as já processadas.

Lê os zips e CSVs de data/raw/ (baixados por scripts/baixar_dados.py) e gera em
data/processed/:

  imagens_256/<base>/<isic_id>.jpg   imagens com o menor lado reduzido para 256 px
                                     (proporção preservada; menores ficam como estão)
  manifesto.csv                      uma linha por imagem, com rótulo, grupo e split
  verificacao.csv                    conferência rótulo x imagem por base
  resumo_por_classe.csv              contagem de imagens por base e classe

Colunas principais do manifesto:
  classe      rótulo harmonizado entre as bases (ver CLASSES abaixo)
  melanoma    1 se a classe é MEL, senão 0
  maligno     1/0 quando dá para afirmar; vazio em AKIEC, INDETERMINADO e UNK
  grupo       paciente (ou lesão, ou a própria imagem) — imagens do mesmo grupo
              nunca são separadas entre treino, validação e teste
  split       train / val / test sugerido (oficial quando a base tem teste oficial)
  duplicada_de  preenchido quando a mesma imagem já aparece em outra linha;
                ao juntar bases, descarte as linhas com esse campo preenchido
"""

import io
import sys
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path(__file__).resolve().parent.parent
RAW = RAIZ / "data" / "raw"
PROC = RAIZ / "data" / "processed"
LADO = 256
SEMENTE = 0

# Classes harmonizadas (vocabulário da ISIC 2019 + extras das outras bases)
#   MEL  melanoma                     NV   nevo melanocítico
#   BCC  carcinoma basocelular        SCC  carcinoma espinocelular (inclui in situ/Bowen e ceratoacantoma)
#   AK   ceratose actínica            AKIEC  AK + Bowen juntos (só ISIC 2018, que não separa os dois)
#   BKL  ceratose benigna (seborreica, lentigo solar, líquen plano-símile)
#   DF   dermatofibroma               VASC lesão vascular
#   INF  inflamatória/infecciosa      OUTRO_BENIGNO / OUTRO_MALIGNO
#   BENIGNO_SEM_DIAGNOSTICO  benigna sem diagnóstico específico (maioria da ISIC 2020 e 2024)
#   INDETERMINADO  proliferação melanocítica atípica
#   UNK  "nenhuma das 8 classes" do teste da ISIC 2019
MALIGNAS = {"MEL", "BCC", "SCC", "OUTRO_MALIGNO"}
BENIGNAS = {"NV", "BKL", "DF", "VASC", "INF", "AK", "OUTRO_BENIGNO", "BENIGNO_SEM_DIAGNOSTICO"}

EXTENSOES = (".jpg", ".jpeg", ".png")


# ----------------------------------------------------------------------------
# Leitura dos rótulos de cada base
# ----------------------------------------------------------------------------

def ler_csv_do_zip(caminho_zip):
    with zipfile.ZipFile(caminho_zip) as z:
        nome = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(nome) as arquivo:
            return pd.read_csv(arquivo)


def rotulo_onehot(df, colunas):
    """Nome da coluna marcada com 1; vazio se a linha não tiver exatamente uma classe."""
    return df[colunas].idxmax(axis=1).where(df[colunas].sum(axis=1) == 1)


def carregar_isic2018():
    pasta = RAW / "isic2018"
    classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    partes = []
    for particao, nome in [("train", "Training"), ("val", "Validation"), ("test", "Test")]:
        gt = ler_csv_do_zip(pasta / f"ISIC2018_Task3_{nome}_GroundTruth.zip")
        partes.append(pd.DataFrame({
            "isic_id": gt["image"],
            "particao_oficial": particao,
            "zip": f"isic2018/ISIC2018_Task3_{nome}_Input.zip",
            "rotulo_original": rotulo_onehot(gt, classes),
        }))
    lesoes = pd.read_csv(pasta / "ISIC2018_Task3_Training_LesionGroupings.csv").rename(
        columns={"image": "isic_id", "lesion_id": "lesao_id", "diagnosis_confirm_type": "confirmacao"})
    df = pd.concat(partes, ignore_index=True).merge(lesoes, on="isic_id", how="left")
    df["classe"] = df["rotulo_original"]
    df["tipo_imagem"] = "dermatoscopia"
    return df


def carregar_isic2019():
    pasta = RAW / "isic2019"
    classes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
    partes = []
    for particao, nome in [("train", "Training"), ("test", "Test")]:
        gt = pd.read_csv(pasta / f"ISIC_2019_{nome}_GroundTruth.csv")
        meta = pd.read_csv(pasta / f"ISIC_2019_{nome}_Metadata.csv")
        gt["rotulo_original"] = rotulo_onehot(gt, classes)
        junto = gt[["image", "rotulo_original"]].merge(meta, on="image", how="left")
        junto["particao_oficial"] = particao
        junto["zip"] = f"isic2019/ISIC_2019_{nome}_Input.zip"
        partes.append(junto)
    df = pd.concat(partes, ignore_index=True).rename(columns={
        "image": "isic_id", "lesion_id": "lesao_id", "age_approx": "idade",
        "sex": "sexo", "anatom_site_general": "local_anatomico"})
    df["classe"] = df["rotulo_original"]
    df["tipo_imagem"] = "dermatoscopia"
    return df


CLASSE_2020 = {
    "melanoma": "MEL",
    "nevus": "NV",
    "seborrheic keratosis": "BKL",
    "lichenoid keratosis": "BKL",
    "solar lentigo": "BKL",
    "lentigo NOS": "BKL",
    "cafe-au-lait macule": "OUTRO_BENIGNO",
    "atypical melanocytic proliferation": "INDETERMINADO",
    "unknown": "BENIGNO_SEM_DIAGNOSTICO",
}


def carregar_isic2020():
    pasta = RAW / "isic2020"
    gt = pd.read_csv(pasta / "ISIC_2020_Training_GroundTruth_v2.csv")
    df = gt.rename(columns={
        "image_name": "isic_id", "patient_id": "paciente_id", "lesion_id": "lesao_id",
        "age_approx": "idade", "sex": "sexo", "anatom_site_general_challenge": "local_anatomico",
        "diagnosis": "rotulo_original"})
    df["classe"] = df["rotulo_original"].map(CLASSE_2020)
    # O gabarito oficial é a coluna target (melanoma sim/não): tem de bater com a classe.
    assert ((df["classe"] == "MEL") == (df["target"] == 1)).all(), "ISIC 2020: classe não bate com target"
    df["particao_oficial"] = "train"
    df["zip"] = "isic2020/ISIC_2020_Training_JPEG.zip"
    df["tipo_imagem"] = "dermatoscopia"

    # Duplicatas declaradas pela própria base: a segunda imagem do par aponta para a primeira.
    pares = pd.read_csv(pasta / "ISIC_2020_Training_Duplicates.csv")
    mapa = dict(zip(pares["image_name_2"], "isic2020:" + pares["image_name_1"]))
    df["duplicada_de"] = df["isic_id"].map(mapa)
    return df.drop(columns=["benign_malignant", "target"])


def classe_isic_dx(d1, d2, d3):
    """Classe harmonizada a partir da hierarquia de diagnóstico da ISIC (diagnosis_1/2/3)."""
    if isinstance(d3, str):
        if d3.startswith("Melanoma"):
            return "MEL"
        if d3.startswith("Squamous cell carcinoma"):
            return "SCC"
        simples = {
            "Basal cell carcinoma": "BCC", "Nevus": "NV", "Dermatofibroma": "DF", "Hemangioma": "VASC",
            "Solar or actinic keratosis": "AK", "Verruca": "INF",
            "Seborrheic keratosis": "BKL", "Solar lentigo": "BKL", "Lichen planus like keratosis": "BKL",
            "Pigmented benign keratosis": "BKL", "Lentigo NOS": "BKL",
            "Atypical intraepithelial melanocytic proliferation": "INDETERMINADO",
            "Atypical melanocytic neoplasm": "INDETERMINADO",
        }
        if d3 in simples:
            return simples[d3]
    if d1 == "Benign":
        # "Benign - Other" sem diagnóstico específico é só "benigna" na avaliação clínica
        return "OUTRO_BENIGNO" if isinstance(d2, str) and d2 != "Benign - Other" else "BENIGNO_SEM_DIAGNOSTICO"
    if d1 == "Malignant":
        return "OUTRO_MALIGNO"
    return "INDETERMINADO" if d1 == "Indeterminate" else None


def carregar_isic2024():
    pasta = RAW / "isic2024"
    gt = pd.read_csv(pasta / "ISIC_2024_Training_GroundTruth.csv")
    sup = pd.read_csv(pasta / "ISIC_2024_Training_Supplement.csv", low_memory=False)
    df = gt.merge(sup, on="isic_id", how="left")
    df["classe"] = [classe_isic_dx(a, b, c) for a, b, c in zip(df["iddx_1"], df["iddx_2"], df["iddx_3"])]
    # O gabarito oficial é a coluna malignant: tem de bater com a classe derivada do diagnóstico.
    assert (df["classe"].isin(MALIGNAS) == (df["malignant"] == 1)).all(), "ISIC 2024: classe não bate com malignant"
    df = df.rename(columns={"lesion_id": "lesao_id", "iddx_full": "rotulo_original"})
    df["particao_oficial"] = "train"
    df["zip"] = "isic2024/ISIC_2024_Training_Input.zip"
    df["tipo_imagem"] = "recorte_foto_3d"
    df = df[["isic_id", "particao_oficial", "zip", "rotulo_original", "classe", "lesao_id", "tipo_imagem"]]

    # Paciente, idade, sexo e local vêm do metadata.csv que está dentro do zip das imagens.
    with zipfile.ZipFile(pasta / "ISIC_2024_Training_Input.zip") as z:
        with z.open("ISIC_2024_Training_Input/metadata.csv") as arquivo:
            meta = pd.read_csv(arquivo, usecols=["isic_id", "patient_id", "age_approx", "sex", "anatom_site_general"])
    meta = meta.rename(columns={"patient_id": "paciente_id", "age_approx": "idade", "sex": "sexo",
                                "anatom_site_general": "local_anatomico"})
    return df.merge(meta, on="isic_id", how="left")


def carregar_milk10k():
    pasta = RAW / "milk10k"
    classes = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
    gt = pd.read_csv(pasta / "MILK10k_Training_GroundTruth.csv")
    gt["rotulo_original"] = rotulo_onehot(gt, classes)
    meta = pd.read_csv(pasta / "MILK10k_Training_Metadata.csv")
    sup = pd.read_csv(pasta / "MILK10k_Training_Supplement.csv")
    df = (meta.merge(sup, on="isic_id", how="left")
              .merge(gt[["lesion_id", "rotulo_original"]], on="lesion_id", how="left"))

    def classe(linha):
        rotulo = linha["rotulo_original"]
        if rotulo == "AKIEC":  # a base junta os dois; o diagnóstico completo separa
            return "AK" if linha["diagnosis_full"] == "Solar or actinic keratosis" else "SCC"
        return {"SCCKA": "SCC", "BEN_OTH": "OUTRO_BENIGNO", "MAL_OTH": "OUTRO_MALIGNO"}.get(rotulo, rotulo)

    df["classe"] = df.apply(classe, axis=1)
    df["tipo_imagem"] = df["image_type"].map({"dermoscopic": "dermatoscopia", "clinical: close-up": "foto_clinica"})
    df["particao_oficial"] = "train"
    df["zip"] = "milk10k/MILK10k_Training_Input.zip"
    df = df.rename(columns={
        "lesion_id": "lesao_id", "age_approx": "idade", "sex": "sexo", "site": "local_anatomico",
        "diagnosis_full": "diagnostico_detalhado", "diagnosis_confirm_type": "confirmacao",
        "skin_tone_class": "tom_de_pele"})
    return df[["isic_id", "particao_oficial", "zip", "rotulo_original", "classe", "lesao_id", "tipo_imagem",
               "idade", "sexo", "local_anatomico", "diagnostico_detalhado", "confirmacao", "tom_de_pele"]]


def carregar_isic_clinicas():
    """Fotos clínicas (celular/câmera) do acervo da ISIC que não estão no treino da MILK10k."""
    pasta = RAW / "isic_clinicas"
    df = pd.read_csv(pasta / "metadados.csv")
    milk = pd.read_csv(RAW / "milk10k" / "MILK10k_Training_Metadata.csv")
    df = df[~df["isic_id"].isin(milk["isic_id"])].copy()
    # As 479 fotos do MILK10k Benchmark (teste cego) vêm sem diagnóstico: não servem para treino.
    sem_rotulo = df["diagnosis_1"].isna()
    print(f"  fotos clínicas sem diagnóstico na ISIC (descartadas): {int(sem_rotulo.sum())}")
    df = df[~sem_rotulo].copy()
    df["classe"] = [classe_isic_dx(a, b, c) for a, b, c in zip(df["diagnosis_1"], df["diagnosis_2"], df["diagnosis_3"])]
    df["rotulo_original"] = df["diagnosis_3"].fillna(df["diagnosis_2"]).fillna(df["diagnosis_1"])
    df["fonte"] = df["attribution"].str.replace("Federal University of Esp\u00edrito Santo (UFES)", "PAD-UFES-20", regex=False)
    df["tipo_imagem"] = df["image_type"].map({"clinical: close-up": "foto_clinica", "clinical: overview": "foto_clinica_longe"})
    df["particao_oficial"] = "train"
    df["zip"] = "isic_clinicas/imagens"
    df = df.rename(columns={"patient_id": "paciente_id", "lesion_id": "lesao_id", "age_approx": "idade", "sex": "sexo",
                            "diagnosis_confirm_type": "confirmacao", "skin_tone_class": "tom_de_pele"})
    df["local_anatomico"] = df["anatom_site_general"].fillna(df["anatom_site_1"])
    return df[["isic_id", "particao_oficial", "zip", "rotulo_original", "classe", "paciente_id", "lesao_id", "tipo_imagem",
               "idade", "sexo", "local_anatomico", "confirmacao", "tom_de_pele", "fonte", "copyright_license"]]


def carregar_mednode():
    """MED-NODE (Groningen): 70 melanomas e 100 nevos em foto clínica; o rótulo é a pasta."""
    caminho = RAW / "mednode" / "complete_mednode_dataset.zip"
    nomes = imagens_do_zip(caminho)
    df = pd.DataFrame({"isic_id": ["mednode_" + Path(n).stem for n in nomes],
                       "rotulo_original": [Path(n).parent.name for n in nomes]})
    df["classe"] = df["rotulo_original"].map({"melanoma": "MEL", "naevus": "NV"})
    df["particao_oficial"] = "train"
    df["zip"] = "mednode/complete_mednode_dataset.zip"
    df["tipo_imagem"] = "foto_clinica"
    return df


# Ordem = prioridade ao marcar duplicatas entre bases (a primeira que tiver a imagem fica com ela).
# Condições da SCIN que correspondem a classes do nosso vocabulário de lesões; o resto é
# inflamatório/infeccioso (INF), que é o foco da base.
SCIN_PARA_CLASSE = {
    "Melanoma": "MEL", "Melanocytic Nevus": "NV", "Basal Cell Carcinoma": "BCC", "SCC/SCCIS": "SCC",
    "Actinic Keratosis": "AK", "Seborrheic Keratosis": "BKL", "Lentigo": "BKL", "Solar Lentigo": "BKL",
    "Dermatofibroma": "DF", "Hemangioma": "VASC", "Cherry Angioma": "VASC", "Pyogenic Granuloma": "VASC",
    "Epidermal Cyst": "OUTRO_BENIGNO", "Lipoma": "OUTRO_BENIGNO", "Skin Tag": "OUTRO_BENIGNO",
    "Keloid": "OUTRO_BENIGNO", "Scar": "OUTRO_BENIGNO",
}
SCIN_IDADE = {"AGE_18_TO_29": 24, "AGE_30_TO_39": 35, "AGE_40_TO_49": 45, "AGE_50_TO_59": 55,
              "AGE_60_TO_69": 65, "AGE_70_TO_79": 75, "AGE_80_OR_ABOVE": 85}


def carregar_scin():
    """SCIN (Google, 2024): fotos de celular de condições de pele enviadas por usuários nos EUA,
    rotuladas por 1 a 3 dermatologistas como diagnóstico diferencial ponderado (rótulo fraco).
    Uma linha por imagem; o grupo é o caso. Casos com foto julgada ilegível ficam INDETERMINADO."""
    import ast
    pasta = RAW / "scin"
    casos = pd.read_csv(pasta / "scin_cases.csv", low_memory=False)
    rotulos = pd.read_csv(pasta / "scin_labels.csv", low_memory=False)
    df = casos.merge(rotulos, on="case_id")
    linhas = []
    for _, c in df.iterrows():
        legivel = str(c["dermatologist_gradable_for_skin_condition_1"]).startswith(("DEFAULT_YES", "YES"))
        pesos = ast.literal_eval(c["weighted_skin_condition_label"]) if isinstance(c["weighted_skin_condition_label"], str) else {}
        if legivel and pesos:
            principal = max(pesos, key=pesos.get)
            classe = SCIN_PARA_CLASSE.get(principal, "INF")
            rotulo = f"{principal} ({pesos[principal]:.2f})"
        else:
            principal, classe, rotulo = None, "INDETERMINADO", "foto ilegível ou sem patologia visível"
        partes = [col.replace("body_parts_", "").lower() for col in casos.columns
                  if col.startswith("body_parts_") and str(c[col]).upper() in ("TRUE", "YES", "1")]
        fst = next((c[f"dermatologist_fitzpatrick_skin_type_label_{i}"] for i in (1, 2, 3)
                    if isinstance(c.get(f"dermatologist_fitzpatrick_skin_type_label_{i}"), str)), pd.NA)
        for i, enquadramento in ((1, "image_1_shot_type"), (2, "image_2_shot_type"), (3, "image_3_shot_type")):
            caminho = c[f"image_{i}_path"]
            if not isinstance(caminho, str):
                continue
            linhas.append({
                "isic_id": "scin_" + Path(caminho).stem, "paciente_id": str(c["case_id"]),
                "classe": classe, "rotulo_original": rotulo,
                "diagnostico_detalhado": c["weighted_skin_condition_label"] if pesos else pd.NA,
                "confirmacao": "diferencial de dermatologistas pela foto" if legivel and pesos else "qualidade insuficiente",
                "idade": SCIN_IDADE.get(c["age_group"], np.nan),
                "sexo": {"FEMALE": "female", "MALE": "male"}.get(c["sex_at_birth"], pd.NA),
                "local_anatomico": ", ".join(partes) if partes else pd.NA,
                "tom_de_pele": fst,
                "tipo_imagem": "foto_celular",
                "fonte": "SCIN", "copyright_license": "SCIN Data Use License",
                "particao_oficial": "train", "zip": "scin/imagens",
            })
    saida = pd.DataFrame(linhas)
    # O README da SCIN avisa: 15 imagens aparecem em mais de um caso (42 ocorrências). Fica a primeira.
    repetidas = saida["isic_id"].duplicated(keep="first")
    if repetidas.any():
        print(f"  SCIN: {int(repetidas.sum())} ocorrências repetidas da mesma imagem descartadas", flush=True)
    return saida[~repetidas].reset_index(drop=True)


BASES = {
    "scin": carregar_scin,
    "isic2019": carregar_isic2019,
    "isic2020": carregar_isic2020,
    "isic2024": carregar_isic2024,
    "milk10k": carregar_milk10k,
    "isic2018": carregar_isic2018,
    "isic_clinicas": carregar_isic_clinicas,
    "mednode": carregar_mednode,
}

# Como sortear o split em cada base: {particao_oficial: {split: nº de vigésimos}}; o resto vira train.
SPLITS = {
    "scin": {"train": {"test": 3, "val": 3}},
    "isic2018": None,  # usa treino/validação/teste oficiais
    "isic2019": {"train": {"val": 3}},  # teste oficial fica como teste
    "isic2020": {"train": {"test": 3, "val": 3}},
    "isic2024": {"train": {"test": 3, "val": 3}},
    "milk10k": {"train": {"test": 3, "val": 3}},
    "isic_clinicas": {"train": {"test": 3, "val": 3}},
    "mednode": {"train": {"test": 3, "val": 3}},
}


# ----------------------------------------------------------------------------
# Redimensionamento
# ----------------------------------------------------------------------------

_ZIPS = {}  # um zip aberto por processo


def ler_bytes(origem, nome):
    """Lê uma imagem de um zip (conferindo o CRC) ou de uma pasta."""
    if Path(origem).is_dir():
        return (Path(origem) / nome).read_bytes()
    if origem not in _ZIPS:
        _ZIPS[origem] = zipfile.ZipFile(origem)
    return _ZIPS[origem].read(nome)


def redimensionar_lote(origem, pasta_saida, nomes, lado, prefixo=""):
    resultados = []
    for nome in nomes:
        isic_id = prefixo + Path(nome).stem
        destino = Path(pasta_saida) / f"{isic_id}.jpg"
        try:
            # Retomada: se a saída já existe e é uma cópia fiel (lado menor < lado, logo não foi
            # redimensionada), reaproveita; só o cabeçalho é lido.
            if destino.exists() and destino.stat().st_size > 0:
                with Image.open(destino) as pronta:
                    largura, altura = pronta.size
                if min(largura, altura) < lado:
                    resultados.append((isic_id, largura, altura, ""))
                    continue
            dados = ler_bytes(origem, nome)
            imagem = Image.open(io.BytesIO(dados))
            largura, altura = imagem.size
            menor = min(largura, altura)
            if menor <= lado and imagem.format == "JPEG" and imagem.mode == "RGB":
                imagem.load()  # decodifica inteira para garantir que não está corrompida
                destino.write_bytes(dados)
            else:
                escala = min(1.0, lado / menor)
                novo = (max(1, round(largura * escala)), max(1, round(altura * escala)))
                if imagem.format == "JPEG":
                    imagem.draft("RGB", novo)  # decodifica já reduzida: muito mais rápido
                imagem = imagem.convert("RGB")
                if imagem.size != novo:
                    imagem = imagem.resize(novo, Image.LANCZOS)
                imagem.save(destino, "JPEG", quality=92)
            resultados.append((isic_id, largura, altura, ""))
        except Exception as erro:
            resultados.append((isic_id, 0, 0, repr(erro)))
    return resultados


def imagens_do_zip(origem):
    """Nomes das imagens dentro de um zip ou de uma pasta (relativos à origem)."""
    origem = Path(origem)
    if origem.is_dir():
        nomes = [str(p.relative_to(origem)) for p in origem.rglob("*")]
    else:
        with zipfile.ZipFile(origem) as z:
            nomes = z.namelist()
    return sorted(n for n in nomes
                  if n.lower().endswith(EXTENSOES) and "__MACOSX" not in n and not Path(n).name.startswith("."))


def processar_zip(executor, base, zip_relativo, prefixo=""):
    """Redimensiona todas as imagens de um zip ou pasta. Devolve DataFrame isic_id, largura, altura, erro."""
    cache = PROC / "_cache" / f"{base}_{Path(zip_relativo).stem}_{LADO}.csv"  # com a base: "imagens" se repete
    if cache.exists():
        return pd.read_csv(cache, keep_default_na=False)

    caminho_zip = str(RAW / zip_relativo)
    pasta_saida = PROC / f"imagens_{LADO}" / base
    pasta_saida.mkdir(parents=True, exist_ok=True)
    nomes = imagens_do_zip(caminho_zip)
    tamanho_lote = 64 if len(nomes) < 100_000 else 512
    lotes = [nomes[i:i + tamanho_lote] for i in range(0, len(nomes), tamanho_lote)]
    resultados = []
    for i, lote in enumerate(executor.map(redimensionar_lote, [caminho_zip] * len(lotes),
                                          [str(pasta_saida)] * len(lotes), lotes, [LADO] * len(lotes),
                                          [prefixo] * len(lotes))):
        resultados.extend(lote)
        if i % max(1, len(lotes) // 10) == 0:
            print(f"  {Path(zip_relativo).name}: {len(resultados)}/{len(nomes)}", flush=True)

    df = pd.DataFrame(resultados, columns=["isic_id", "largura", "altura", "erro"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


# ----------------------------------------------------------------------------
# Grupos e splits
# ----------------------------------------------------------------------------

def sortear_split(df, vigesimos):
    """Divide mantendo cada grupo inteiro num split só e a proporção das classes parecida."""
    divisor = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEMENTE)
    dobra = np.zeros(len(df), dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # classes com menos de 20 imagens geram aviso; é esperado
        for i, (_, indices) in enumerate(divisor.split(df, df["classe"], df["grupo"])):
            dobra[indices] = i
    split = np.full(len(df), "train", dtype=object)
    inicio = 0
    for nome, quantidade in vigesimos.items():
        split[(dobra >= inicio) & (dobra < inicio + quantidade)] = nome
        inicio += quantidade
    return split


def atribuir_splits(df, base):
    df["split"] = df["particao_oficial"]
    regras = SPLITS[base]
    if regras:
        for particao, vigesimos in regras.items():
            mascara = (df["particao_oficial"] == particao).to_numpy()
            df.loc[mascara, "split"] = sortear_split(df[mascara], vigesimos)
    return df


# ----------------------------------------------------------------------------
# Principal
# ----------------------------------------------------------------------------

COLUNAS = ["isic_id", "base", "fonte", "split", "classe", "melanoma", "maligno", "arquivo", "tipo_imagem",
           "rotulo_original", "diagnostico_detalhado", "confirmacao", "grupo", "paciente_id", "lesao_id",
           "duplicada_de", "particao_oficial", "idade", "sexo", "local_anatomico", "tom_de_pele",
           "largura", "altura", "copyright_license"]


def preparar_base(executor, base):
    print(f"\n== {base}", flush=True)
    df = BASES[base]()
    df["base"] = base
    for coluna in COLUNAS:
        if coluna not in df.columns:
            df[coluna] = pd.NA
    assert df["isic_id"].is_unique, f"{base}: isic_id repetido nos rótulos"

    verificacao = []
    tamanhos = []
    for zip_relativo, rotulos in df.groupby("zip", sort=False):
        imagens = processar_zip(executor, base, zip_relativo, prefixo={"mednode": "mednode_", "scin": "scin_"}.get(base, ""))
        assert imagens["isic_id"].is_unique, f"{zip_relativo}: nome de imagem repetido"
        com_erro = imagens[imagens["erro"] != ""]
        ids_rotulo, ids_imagem = set(rotulos["isic_id"]), set(imagens["isic_id"])
        verificacao.append({
            "base": base,
            "particao_oficial": rotulos["particao_oficial"].iloc[0],
            "linhas_de_rotulo": len(rotulos),
            "imagens_no_zip": len(imagens),
            "rotulo_sem_imagem": len(ids_rotulo - ids_imagem),
            "imagem_sem_rotulo": len(ids_imagem - ids_rotulo),
            "sem_classe": int(rotulos["classe"].isna().sum()),
            "imagem_corrompida": len(com_erro),
        })
        for _, linha in com_erro.iterrows():
            print(f"  ERRO ao abrir {linha['isic_id']}: {linha['erro']}")
        tamanhos.append(imagens[imagens["erro"] == ""][["isic_id", "largura", "altura"]])

    df = df.drop(columns=["largura", "altura"]).merge(pd.concat(tamanhos), on="isic_id", how="left")
    tem_imagem = df["largura"].notna()
    df["arquivo"] = (f"imagens_{LADO}/{base}/" + df["isic_id"] + ".jpg").where(tem_imagem)

    df["melanoma"] = (df["classe"] == "MEL").astype(int).where(df["classe"].notna())
    df["maligno"] = np.select([df["classe"].isin(MALIGNAS), df["classe"].isin(BENIGNAS)], [1, 0], default=np.nan)

    # Grupo: paciente quando a base informa, senão lesão, senão a própria imagem.
    identificador = df["paciente_id"].fillna(df["lesao_id"]).fillna(df["isic_id"])
    df["grupo"] = base + ":" + identificador.astype(str)
    df["fonte"] = df["fonte"].fillna(base)
    df = atribuir_splits(df, base)
    df = df[COLUNAS]
    df.to_csv(PROC / "_cache" / f"manifesto_{base}.csv", index=False)
    pd.DataFrame(verificacao).to_csv(PROC / "_cache" / f"verificacao_{base}.csv", index=False)
    return df, verificacao


def main():
    bases = sys.argv[1:] or list(BASES)
    PROC.mkdir(parents=True, exist_ok=True)

    (PROC / "_cache").mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=6) as executor:
        for base in [b for b in BASES if b in bases]:
            preparar_base(executor, base)

    # Junta todas as bases já processadas (nesta rodada ou em rodadas anteriores).
    prontas = [b for b in BASES if (PROC / "_cache" / f"manifesto_{b}.csv").exists()]
    manifesto = pd.concat([pd.read_csv(PROC / "_cache" / f"manifesto_{b}.csv", low_memory=False) for b in prontas],
                          ignore_index=True)
    verificacao = pd.concat([pd.read_csv(PROC / "_cache" / f"verificacao_{b}.csv") for b in prontas], ignore_index=True)
    print(f"\nBases no manifesto: {prontas}; faltam: {[b for b in BASES if b not in prontas]}")

    # Mesma imagem em mais de uma base (ex.: a ISIC 2018 inteira está dentro da 2019).
    id_limpo = manifesto["isic_id"].str.replace("_downsampled", "", regex=False)
    primeira = manifesto.groupby(id_limpo)["base"].transform("first")
    repetida = id_limpo.duplicated(keep="first")
    manifesto.loc[repetida, "duplicada_de"] = manifesto.loc[repetida, "duplicada_de"].fillna(
        primeira[repetida] + ":" + id_limpo[repetida])

    manifesto = manifesto[COLUNAS]
    manifesto.to_csv(PROC / "manifesto.csv", index=False)

    verificacao.to_csv(PROC / "verificacao.csv", index=False)
    resumo = pd.crosstab(manifesto["classe"].fillna("SEM_ROTULO"), manifesto["base"], margins=True, margins_name="TOTAL")
    resumo.to_csv(PROC / "resumo_por_classe.csv")

    pd.set_option("display.width", 250)
    print("\n== Verificação rótulo x imagem\n" + verificacao.to_string(index=False))
    print("\n== Imagens por classe e base (todas as linhas)\n" + resumo.to_string())
    print(f"\nLinhas sem arquivo de imagem: {int(manifesto['arquivo'].isna().sum())}")
    print(f"Linhas marcadas como duplicadas: {int(manifesto['duplicada_de'].notna().sum())}")
    print(f"Manifesto: {PROC / 'manifesto.csv'} ({len(manifesto)} linhas)")


if __name__ == "__main__":
    main()

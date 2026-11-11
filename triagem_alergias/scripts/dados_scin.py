"""Tabela do especialista de erupções: split do Felipe (splits/scin_v1.csv) + imagens da SCIN baixada por
scripts/baixar_scin.py + rótulos suaves e metadados do paciente reconstruídos dos CSVs da SCIN.

Reproduz as mesmas regras do triagem_alergias.ipynb (sinônimos somados, condição principal = maior peso,
classes = 16 condições com >= 30 casos + outras_condicoes) para que os números sejam comparáveis.

Uso como módulo:
    from dados_scin import tabela_especialista
    df, classes, alvos_suaves, meta = tabela_especialista()
"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
PASTA = RAIZ / "triagem_alergias"
SCIN_RAW = RAIZ / "data" / "raw" / "scin"
IMAGENS = RAIZ / "data" / "processed" / "imagens_256" / "scin"   # lado menor 256 px, JPEG 92 (preparar_dados.py)
SPLIT = PASTA / "splits" / "scin_v1.csv"
OTHER = "outras_condicoes"
SYNONYMS = {
    "Allergic Contact Dermatitis": "Contact Dermatitis",
    "Irritant Contact Dermatitis": "Contact Dermatitis",
    "CD - Contact dermatitis": "Contact Dermatitis",
    "Acute dermatitis, NOS": "Dermatitis, NOS",
    "Chronic dermatitis, NOS": "Dermatitis, NOS",
    "Acute and chronic dermatitis": "Dermatitis, NOS",
}
# metadados autodeclarados do paciente (todos booleanos YES/NO ou categóricos), sem raça
META_BOOL = [
    "textures_raised_or_bumpy", "textures_flat", "textures_rough_or_flaky", "textures_fluid_filled",
    "body_parts_head_or_neck", "body_parts_arm", "body_parts_palm", "body_parts_back_of_hand", "body_parts_torso_front",
    "body_parts_torso_back", "body_parts_genitalia_or_groin", "body_parts_buttocks", "body_parts_leg",
    "body_parts_foot_top_or_side", "body_parts_foot_sole", "body_parts_other",
    "condition_symptoms_bothersome_appearance", "condition_symptoms_bleeding", "condition_symptoms_increasing_size",
    "condition_symptoms_darkening", "condition_symptoms_itching", "condition_symptoms_burning", "condition_symptoms_pain",
    "condition_symptoms_no_relevant_experience", "other_symptoms_fever", "other_symptoms_chills", "other_symptoms_fatigue",
    "other_symptoms_joint_pain", "other_symptoms_mouth_sores", "other_symptoms_shortness_of_breath",
    "other_symptoms_no_relevant_symptoms",
]
META_CAT = ["age_group", "sex_at_birth", "fitzpatrick_skin_type", "related_category", "condition_duration"]


def _pesos(texto):
    d = ast.literal_eval(texto) if isinstance(texto, str) and texto.strip() else {}
    out = {}
    for k, v in d.items():
        k = SYNONYMS.get(k, k)
        out[k] = out.get(k, 0.0) + float(v)
    return out


def tabela_especialista():
    """DataFrame (uma linha por imagem do grupo erupcao_alergia do split v1), lista de classes,
    matriz N x classes de rótulos suaves e DataFrame de metadados one-hot alinhado."""
    split = pd.read_csv(SPLIT, dtype={"image_id": str, "case_id": str})
    split = split[split["grupo"] == "erupcao_alergia"].reset_index(drop=True)
    casos = pd.read_csv(SCIN_RAW / "scin_cases.csv", dtype=str, low_memory=False)
    rotulos = pd.read_csv(SCIN_RAW / "scin_labels.csv", dtype=str, low_memory=False)
    casos = casos.merge(rotulos[["case_id", "weighted_skin_condition_label"]], on="case_id", how="left")
    casos["pesos"] = casos["weighted_skin_condition_label"].map(_pesos)
    df = split.merge(casos[["case_id", "pesos", *META_BOOL, *META_CAT]], on="case_id", how="left", validate="m:1")
    assert df["pesos"].notna().all(), "caso do split sem rótulo na scin_labels.csv"
    df["arquivo"] = [str(IMAGENS / f"scin_{i}.jpg") for i in df["image_id"]]
    faltam = [a for a in df["arquivo"] if not Path(a).exists()]
    assert not faltam, f"{len(faltam)} imagens do split não estão em {IMAGENS}: {faltam[:3]}"

    # mesma ordem de classes do notebook: ordenadas + outras no fim; confere com o split
    classes = sorted(c for c in df["classe"].unique() if c != OTHER) + [OTHER]
    idx = {c: i for i, c in enumerate(classes)}

    def suave(pesos):
        t = np.zeros(len(classes), "float32")
        for cond, w in pesos.items():
            t[idx.get(cond, idx[OTHER])] += w
        return t / t.sum()

    alvos = np.stack(df["pesos"].map(suave).values)
    # conferência: a condição principal reconstruída bate com a classe do split em quase tudo
    # (empates podem diferir; o split do Felipe é a referência)
    concord = (alvos.argmax(1) == df["classe"].map(idx).values).mean()
    df.attrs["concordancia_rotulo"] = float(concord)

    meta = pd.get_dummies(df[META_CAT].fillna("NA"), dtype="float32")
    for c in META_BOOL:
        meta[c] = df[c].astype(str).str.upper().isin(["YES", "TRUE", "1"]).astype("float32")
    meta.index = df.index
    return df, classes, alvos, meta


if __name__ == "__main__":
    df, classes, alvos, meta = tabela_especialista()
    print(len(df), "imagens,", df["case_id"].nunique(), "casos,", len(classes), "classes,", meta.shape[1], "metadados")
    print("concordância da classe principal reconstruída com o split:", round(df.attrs["concordancia_rotulo"], 4))
    print(df["split"].value_counts().to_dict())
    print(json.dumps({s: df[df["split"] == s]["classe"].value_counts().to_dict() for s in ["test"]}, ensure_ascii=False, indent=1))

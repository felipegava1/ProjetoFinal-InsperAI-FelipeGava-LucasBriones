"""Lista na API da ISIC todas as fotos clínicas (celular/câmera, sem dermatoscópio).

Uso:  py -3.12 scripts/baixar_metadados_clinicas.py
Saída: data/raw/isic_clinicas/metadados.csv (uma linha por imagem, com rótulo, coleção e URL)
"""
import csv, json, time, urllib.parse, urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "data" / "raw" / "isic_clinicas" / "metadados.csv"
API = "https://api.isic-archive.com/api/v2/images/search/?"
CAMPOS_CLINICOS = ["patient_id", "lesion_id", "diagnosis_1", "diagnosis_2", "diagnosis_3", "diagnosis_4", "diagnosis_5",
                   "diagnosis_confirm_type", "benign_malignant", "age_approx", "sex", "anatom_site_general", "anatom_site_1",
                   "concomitant_biopsy", "fitzpatrick_skin_type", "skin_tone_class"]
CAMPOS_AQUISICAO = ["image_type", "image_manipulation", "pixels_x", "pixels_y"]

def pedir(url):
    for tentativa in range(1, 8):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except Exception as erro:
            print(f"falha {tentativa}: {erro}", flush=True); time.sleep(5 * tentativa)
    raise SystemExit("API não respondeu")

def main():
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    linhas = []
    for tipo in ["clinical: close-up", "clinical: overview"]:
        url = API + urllib.parse.urlencode({"query": f'image_type:"{tipo}"', "limit": 100})
        while url:
            pagina = pedir(url)
            for im in pagina["results"]:
                cl, aq = im["metadata"].get("clinical", {}), im["metadata"].get("acquisition", {})
                linhas.append({
                    "isic_id": im["isic_id"], "copyright_license": im.get("copyright_license"),
                    "attribution": im.get("attribution"), "public": im.get("public"),
                    **{c: cl.get(c, "") for c in CAMPOS_CLINICOS}, **{c: aq.get(c, "") for c in CAMPOS_AQUISICAO},
                    "url_full": im["files"]["full"]["url"], "bytes_full": im["files"]["full"]["size"],
                    "url_256": im["files"].get("thumbnail_256", {}).get("url", ""),
                })
            url = pagina.get("next")
            if len(linhas) % 2000 < 100:
                print(f"{len(linhas)} de {pagina['count']} ({tipo})", flush=True)
    with open(SAIDA, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0])); w.writeheader(); w.writerows(linhas)
    print(f"{len(linhas)} imagens -> {SAIDA}")

if __name__ == "__main__":
    main()

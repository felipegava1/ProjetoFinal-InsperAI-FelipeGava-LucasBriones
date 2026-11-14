"""Gera notebook_dermatoscopia.ipynb (registro das iterações E0–E5) a partir dos resultados salvos.

Uso:
    python scripts_dermato/gerar_notebook.py            # gera e executa (só lê resultados; CPU, ~1 min)
    python scripts_dermato/gerar_notebook.py --sem-executar

Rode de novo quando uma etapa nova terminar: as seções de E3, E4 e E5 aparecem com os números assim
que os modelos finais estiverem em experimentos.csv. Os textos de cada seção resumem o DIARIO.md.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import nbformat as nbf
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SAIDA = REPO / "notebook_dermatoscopia.ipynb"
cells = []


def md(texto):
    cells.append(nbf.v4.new_markdown_cell(texto.strip("\n")))


def code(texto):
    cells.append(nbf.v4.new_code_cell(texto.strip("\n")))


def secao_busca(nome, titulo, como_funciona, variado_extra, funcionou):
    md(f"## {titulo}\n\n### 1. Como funciona\n\n{como_funciona}")
    md(f"### 2. O que foi variado\n\n{variado_extra}")
    code(f"faixas, melhores, vencedora = rel.resumo_busca('{nome}')\nfaixas")
    md("### 3. Resultados\n\nAs 5 melhores tentativas na validação e a configuração vencedora:")
    code("display(melhores)\nprint('Configuração vencedora:', vencedora)")
    code(f"fig, axs = plt.subplots(1, 2, figsize=(12, 3.4))\nrel.grafico_busca('{nome}', axs[0])\n"
         f"rel.grafico_importancia('{nome}', axs[1])\nplt.tight_layout(); plt.show()")
    md(f"### 4. O que funcionou, o que não e por quê\n\n{funcionou}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem-executar", action="store_true")
    a = ap.parse_args()
    exp = pd.read_csv(REPO / "resultados" / "experimentos.csv")
    tem = set(exp["id"])
    etapas = set(exp["etapa"])

    md("""
# Classificador de dermatoscopia — registro das iterações (E0 → E5)

**Problema:** apoiar dermatologistas e serviços de teledermatologia a priorizar lesões suspeitas em
imagens de dermatoscopia, reduzindo encaminhamentos e biópsias desnecessárias sem deixar passar casos
de câncer. O resultado é **apoio à priorização, nunca diagnóstico** (o diagnóstico é a histopatologia).

Este notebook **não treina nada**: lê `resultados/experimentos.csv` (uma linha por treino),
`resultados/optuna.db` (todas as tentativas das buscas) e as previsões de validação salvas em
`resultados/previsoes/`. Roda em qualquer PC com o repositório, sem as imagens e sem GPU. O raciocínio
completo, na ordem em que aconteceu, está em `resultados/DIARIO.md`.

> **Testes T1/T2 ainda fechados.** Tudo abaixo é validação. Pelo protocolo, os testes são rodados uma
> única vez, nos finalistas, no fim.
""")
    code("""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from dermato import relatorio as rel
from dermato.config import CLASSES
rel.estilo()
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
""")

    # ------------------------------------------------------------------ dados e protocolo
    md("""
## Dados e protocolo comum

Divisão fixa (`splits/dermato.csv`, semente 42), feita uma vez e nunca alterada. Validação = 15% do
conjunto de treino, sorteada **por grupo** (paciente ou lesão), estratificada por classe.
""")
    code("""
sp = pd.read_csv('splits/dermato.csv')
pd.crosstab(sp['base'], sp['split'], margins=True)
""")
    code("pd.crosstab(sp['dx'].map(rel.NOMES), sp['split'], margins=True)")
    md("""
| Métrica | Papel |
|---|---|
| Acurácia balanceada (validação) | principal: decide quem avança |
| Especificidade a 95% de sensibilidade (**spec95**) | liga o modelo ao problema: com 95% dos malignos pegos, quantos benignos saem da fila prioritária |
| AUC de melanoma, AUC maligno × benigno | secundárias |
| ECE (erro de calibração) | se as probabilidades são confiáveis |

Regras seguidas em todas as etapas: mesma divisão; época, parâmetros, tau (ajuste de prior) e limiar
escolhidos **só na validação**; TTA de 8 vistas em todos; busca automática com Optuna (corte das
tentativas ruins a partir da 4ª época); cada rodada registrada em `experimentos.csv`.

**Ruído entre rodadas** (medido, ver diário): a mesma configuração com outra sequência aleatória de
augmentation varia **~1 ponto de acurácia balanceada** e **~2 pontos de spec95**. Diferenças menores
que isso são empate.
""")

    # ------------------------------------------------------------------ E0
    md("""
## E0 — receita atual (a régua)

### 1. Como funciona
A CNN do modelo anterior, portada de Keras para PyTorch e retreinada só com dermatoscopia:
EfficientNetV2-B0 (ImageNet), 224 px, 14 metadados (idade, sexo, região), Shades of Gray,
augmentation leve, AdamW lr 1e-4, 8 épocas, BatchNorm congelada, nevos limitados a 6.000, entropia
cruzada + ajuste de prior. Não ataca nenhuma fraqueza: é o ponto de partida de todas as comparações.

### 2. O que foi variado
Nada — um treino só, com a receita fixa.
""")
    code("display(rel.tabela_finais(['E0_final_s42'], ['E0 receita']))\nrel.curva('E0_final_s42'); plt.show()")
    md("""
### 3–4. Resultado e leitura
- Acurácia balanceada **0,696**, spec95 **0,689**, AUC melanoma 0,912 (validação, TTA 8).
- **A curva ainda subia na 8ª época**: partindo do ImageNet, 8 épocas são poucas (o Keras original
  partia de pesos já ajustados no ISIC 2019). Primeira hipótese para a busca: mais épocas.
- tau alto (1,45): treinado na distribuição natural (muitos nevos), precisa de desconto forte do prior.
""")

    secao_busca("E0_busca", "E0_busca — a mesma rede com parâmetros otimizados", """
Mesma EfficientNetV2-B0 do E0, agora com todos os parâmetros livres (inclusive resolução 224/288/384).
Separa "ganho de ajustar parâmetros" de "ganho de trocar de rede": a comparação justa de E1/E2 é contra
esta etapa, não só contra a receita.
""", "Todos os parâmetros comuns do plano (faixas abaixo), orçamento de 1,25 h + ~20 min.", """
- **+3,9 pontos de acurácia balanceada (0,696 → 0,7345) só com parâmetros**, mesma rede e resolução.
- lr ~4× maior (3,8e-4), suavização de rótulos 0,11, dropout/drop path maiores e **todos os nevos**
  (mais dados) explicam o ganho; lr < 3e-5 foi ruim em qualquer combinação.
- Lote 16 nas duas melhores (mais passos por época); augmentation leve e pouco mixup ganharam da forte.
- **Sem Shades of Gray** venceu — surpresa; repetiu-se na E1 e na E2.
""")

    secao_busca("E1", "E1 — CNN mais forte (ConvNeXt-Tiny, até 384 px)", """
ConvNeXt-Tiny (ImageNet-22k) com resolução até 384 px. **Fraqueza atacada:** melanoma reconhecido em só
~62% — detalhes finos (rede de pigmento, bordas) se perdem a 224 px. **Hipótese:** mais resolução + CNN
moderna melhoram AUC e acerto de melanoma. A busca foi semeada com a vencedora da E0_busca a 384 px.
""", "Parâmetros comuns + resolução 224/288/384 px; 3,5 h, 16 tentativas.", """
- **Hipótese confirmada.** Acerto em melanoma 0,62 → **0,77** (+14 pontos sobre a E0_busca, 919 melanomas
  na validação); bal_acc 0,769 e spec95 0,773 (+7 pontos sobre a E0_busca: ~7% a mais dos benignos sairiam
  da fila prioritária com a mesma sensibilidade).
- **Resolução é o fator mais forte**: a tentativa 8 tinha a configuração da 2, mas a 224 px — estava em
  0,642 na 4ª época e foi cortada; a 2, a 384 px, chegou a 0,734.
- Perda **focal** com gama ~3 venceu; o gama ficou no teto da faixa (testado depois, na E1r).
- ECE alto (0,31): focal + suavização deixam as probabilidades "achatadas"; as decisões não mudam, e a
  calibração por temperatura (E5) corrige a escala.
""")
    code("rel.curva('E1_final_s42'); plt.show()")

    secao_busca("E2", "E2 — híbrido em uma rede (MaxViT-Tiny, 224 px)", """
MaxViT-Tiny: convolução e atenção em cada estágio. **Fraqueza atacada:** confusão entre ceratose benigna,
ceratose actínica e carcinoma espinocelular, que depende da forma global. **Hipótese:** a atenção capta
assimetria e forma melhor que a convolução. Fixada em 224 px por custo (4 min/época a 384 px).
""", "Parâmetros comuns + decaimento de lr por camada (0,6–1,0); semeada com a vencedora da E1.", """
- bal_acc 0,746–0,749: acima do B0 otimizado (0,7345) na mesma resolução, abaixo da E1 a 384 px.
- **Hipótese não confirmada**: acerta mais ceratose benigna (0,708, melhor de todos), mas a confusão
  bkl/ak/scc fica igual à do B0 (0,180 × 0,179).
- O decaimento por camada não mudou nada mensurável (0,65/0,8/0,95 → 0,746/0,749/0,745).
- Incidente: a VRAM transbordou para a RAM do sistema (épocas 10× mais lentas); corrigido com passos
  menores e teto de memória (ver diário).
""")

    if "E1r" in etapas:
        md("""
## E1r — rodada extra da E1 (refino)

Busca estreitada em volta do que funcionou (sem lr baixo, sem augmentation forte, sem SoG), testando
duas hipóteses diretas: **448 px** (mais resolução ainda ajuda?) e **gama da focal acima de 3**.
""")
        code("faixas, melhores, vencedora = rel.resumo_busca('E1r')\ndisplay(melhores); print(vencedora)")
        md("""
- **448 px piorou** (0,758 × 0,769) e é ~40% mais lento: a família ConvNeXt fica em 384 px.
- Gama 4,5 deu 0,775 na busca, mas **o retreino deu 0,765**: a diferença é ruído de augmentation (o
  número de processos de carregamento mudou entre as duas rodadas). Conclusão correta: gama 3–4,5 empatam.
- Foi daqui que veio a medida de **ruído ≈ 1 ponto** usada em todo o notebook.
""")

    if "E2_384" in etapas:
        secao_busca("E2_384", "E2 a 384 px — arquitetura × resolução", """
A mesma MaxViT na resolução da E1, para separar "a MaxViT é pior" de "224 px é pior". Busca estreitada
e semeada com a vencedora da E2.
""", "Espaço estreitado (faixas abaixo), 5 h, ~1 h por tentativa.", """
- **Resolução explica a maior parte da distância entre E1 e E2**: na própria MaxViT, 224 → 384 px deu
  +10 pontos em melanoma e +6,5 em spec95.
- **Na mesma resolução, as arquiteturas empatam**: bal_acc 0,768 × 0,769, spec95 0,747–0,774 × 0,773.
- A hipótese da E2 segue não confirmada (confusão bkl/ak/scc 0,179); a MaxViT é melhor em carcinoma
  espinocelular, o ConvNeXt em melanoma e dermatofibroma — erram em lesões diferentes (bom para a E5).
""")

    if "PanDerm_rapido" in etapas:
        md("""
## Teste rápido do PanDerm (antes da E4)

Vetores congelados do PanDerm (ViT-L/16, modelo de fundação de dermatologia), sem ajuste fino, com três
classificadores simples na CPU.
""")
        code("""
pr = rel.experimentos().query("etapa == 'PanDerm_rapido'")
pr[['familia','val_bal_acc','val_spec95','val_mel_auc','val_mal_auc','tempo_treino_min']].round(3)
""")
        md("""
- **As AUCs empatam com o E0 receita** (maligno × benigno 0,935 × 0,933) em 1–4 min de CPU, sem ajuste
  fino: um modelo pronto chega perto de uma CNN treinada, a uma fração do custo, na decisão binária.
- Na classificação em 8 classes fica abaixo (0,54–0,63): os vetores separam maligno/benigno, mas
  classificadores simples erram mais nas classes raras. A E4 segue como planejada.
""")

    if "E3" in etapas:
        secao_busca("E3", "E3 — fusão de dois ramos (ConvNeXt + Swin-Tiny)", """
ConvNeXt (detalhe local, partindo dos pesos da E1) e Swin-Tiny (relações entre regiões distantes) em
paralelo, mais metadados; a junção (concatenação, porta aprendida ou atenção cruzada) decide quanto
confiar em cada um. **Hipótese:** dois olhares separados se complementam mais do que misturados numa
rede (E2).
""", "Parâmetros comuns + tipo de junção, lr do ramo Swin (0,1–1× o do ConvNeXt), épocas com os ramos "
             "congelados (0–3), cabeça 256/512; 3 h.", """
- **A fusão não superou a E1 sozinha**: melhor 0,767 × 0,769 (empate dentro do ruído); spec95
  0,756–0,773 × 0,773. Hipótese não confirmada nesta forma: o Swin a 224 px não acrescentou informação
  que o ConvNeXt a 384 px já não tivesse.
- Como o ramo ConvNeXt já parte treinado, **lr baixo (5e-5)** funcionou melhor que o da E1 (2,6e-4).
- 384 px de novo nas melhores; junção por atenção cruzada venceu, mas com poucas tentativas por tipo.
""")

    if "E4" in etapas:
        secao_busca("E4", "E4 — fusão + conhecimento de dermatologia (PanDerm)", """
Terceiro ramo na fusão da E3: o vetor congelado do PanDerm (1.024 números por imagem, sem ajuste fino).
**Fraqueza atacada:** classes raras com poucos exemplos (dermatofibroma, lesão vascular, carcinoma
espinocelular). **Hipótese:** um modelo que viu muito mais pele compensa a falta de exemplos raros. A
busca foi semeada com a vencedora da E3 + PanDerm.
""", "Os mesmos parâmetros da E3 + dropout no vetor do PanDerm (0–0,6) e vetor normalizado ou não; 3 h.", """
- **O PanDerm não acrescentou nada**: a vencedora (a semeada) deu 0,766 / spec95 0,756; a mesma
  configuração sem o PanDerm (E3 #7) deu 0,767 / 0,756 — idêntico. Retreino: 0,757 / 0,749.
- **Hipótese não confirmada**: os vetores congelados repetem informação que a CNN treinada já tem; o teste
  rápido já indicava que eles separam maligno/benigno bem, mas não as classes raras.
- Efeito colateral bom: as melhores tentativas tiveram calibração excelente (ECE 0,024–0,036).
""")

    if "E5" in etapas:
        md("""
## E5 — ensemble final com calibração

### 1. Como funciona
Combina as previsões de validação da melhor de cada família (E1, E2 a 384 px, E3, E4) e calibra as
probabilidades com uma temperatura. **Fraqueza atacada:** variação entre modelos e probabilidades mal
calibradas (ECE de ~0,3 em E1/E2). **Hipótese:** modelos diferentes erram em imagens diferentes. Como usa as
previsões da própria validação, é **ajustado na metade A** e **avaliado na metade B** (divisão por grupo).

### 2. O que foi variado
Todos os subconjuntos dos 4 modelos × média simples, média ponderada (pesos ótimos na metade A) e LightGBM
sobre as probabilidades; temperatura ajustada na metade A. Minutos na CPU.
""")
        code("""
ens = pd.read_csv('resultados/ensemble_busca.csv')
ens[['membros','metodo','bal_acc','spec95','mel_auc','mal_auc','ece','temperatura']].head(10).round(4)
""")
        md("""
### 3–4. Resultado e leitura (metade B da validação)
- Vencedor: **média simples dos 4 modelos + temperatura**: bal_acc 0,775, **spec95 0,775, AUC melanoma
  0,945, AUC maligno 0,950, ECE 0,056**. A E1 sozinha, na mesma metade B: 0,769 / 0,757 / 0,929 / 0,930 / 0,190.
- O ganho em bal_acc (+0,6) está dentro do ruído; **os ganhos reais são de ordenação e calibração**
  (AUC maligno +0,020, AUC melanoma +0,016, spec95 +1,8 ponto, ECE 0,19 → 0,056): exatamente o que importa
  para triagem. **Hipótese confirmada** nas métricas de triagem.
- Média simples ganhou da ponderada e do LightGBM: com metade da validação para ajustar, mais parâmetros
  não compensam.
- Achado: a decisão em 8 classes da E2 a 384 px é muito sensível ao tau (0,722 na metade B com o tau da
  metade A, 0,779 com o da validação inteira); a média do ensemble amortece essa instabilidade.
""")

    sementes = [f for f in ("E1", "E2_384", "E3") if f"{f}_final_s44" in tem]
    if sementes:
        rot_s = {"E1": "E1 ConvNeXt 384", "E2_384": "E2 MaxViT 384", "E3": "E3 fusão ConvNeXt+Swin"}
        md("""
## Finalistas: 3 sementes cada

O ruído entre rodadas (~1 ponto) é do tamanho das diferenças entre os melhores modelos. Os 3 maiores
bal_acc da validação foram retreinados com as sementes 43 e 44 (mesma configuração), como manda o
protocolo: **uma diferença menor que a variação entre sementes é empate.**
""")
        code(f"display(rel.tabela_sementes({json.dumps(sementes)}, {json.dumps([rot_s[f] for f in sementes])}))\n"
             f"rel.grafico_sementes({json.dumps(sementes)}, {json.dumps([rot_s[f] for f in sementes])}); "
             "plt.tight_layout(); plt.show()")
        md("""
- **Acurácia balanceada: empate.** E1 0,764 ± 0,005, E3 0,761 ± 0,003, E2 a 384 px 0,757 ± 0,010 — as
  médias diferem menos que o desvio entre sementes.
- **Especificidade a 95%: vantagem pequena e consistente da E1** (0,766 ± 0,006 × 0,754 ± 0,008 da E3 e
  0,746 ± 0,004 da E2 a 384 px): as três sementes da E1 ficam acima de todas as da E2 a 384 px.
- AUCs praticamente iguais (melanoma 0,922–0,934; maligno 0,947–0,952).
- Os números de uma semente só (seção de cada etapa) são um pouco otimistas: a seed 42 foi a escolhida pela
  busca. A média de 3 sementes é a estimativa honesta para levar aos testes.
""")

    # ------------------------------------------------------------------ linha de raciocínio
    finais = [("E0_final_s42", "E0 receita (B0 224)"), ("E0_busca_final_s42", "E0_busca (B0 224)"),
              ("E2_final_s42", "E2 MaxViT 224"), ("E2_384_final_s42", "E2 MaxViT 384"),
              ("E1_final_s42", "E1 ConvNeXt 384"), ("E1r_final_s42", "E1r ConvNeXt 384 (refino)"),
              ("E3_final_s42", "E3 fusão ConvNeXt+Swin"), ("E4_final_s42", "E4 fusão + PanDerm"),
              ("E5_ensemble", "E5 ensemble (metade B)")]
    finais = [(i, r) for i, r in finais if i in tem]
    md("""
## Linha de raciocínio

1. **E0 → E0_busca:** a receita estava sub-treinada (curva subindo na última época); otimizar parâmetros
   na mesma rede deu +3,9 pontos. Isso virou a régua justa para as redes novas.
2. **E0_busca → E1:** a fraqueza medida era melanoma (~62%). ConvNeXt a 384 px levou melanoma a 77% e spec95
   a 0,773 — maior ganho do projeto. A resolução, mais que a rede, explicou o salto.
3. **E1 → E2:** sobrava a confusão entre ceratoses e carcinoma espinocelular. A MaxViT não a reduziu; a
   224 px ficou atrás da E1, e a 384 px empatou com ela — a diferença era resolução, não arquitetura.
4. **E1/E2 → E3:** juntar CNN e Transformer em ramos separados não superou a CNN sozinha: o ramo Swin
   (224 px) não trouxe informação nova para um ConvNeXt já a 384 px.
5. **E3 → E4:** o PanDerm congelado separava maligno/benigno tão bem quanto o E0, mas como terceiro ramo
   não acrescentou nada: repete informação que a CNN treinada já tem.
6. **E1–E4 → E5:** como os modelos erram em lesões diferentes (MaxViT melhor em carcinoma espinocelular,
   ConvNeXt em melanoma e dermatofibroma), a média deles com calibração melhorou ordenação (AUC maligno
   0,930 → 0,950) e calibração (ECE 0,19 → 0,056) — o que importa para priorizar a fila.
7. **Finalistas:** com 3 sementes, E1, E3 e E2 a 384 px empatam em acurácia balanceada; a E1 mantém
   uma vantagem pequena em spec95.

A tabela abaixo usa uma semente por modelo (a 42); a linha do E5 é medida só na metade B da validação.
""")
    lista = json.dumps([i for i, _ in finais])
    rots = json.dumps([r for _, r in finais])
    code(f"finais = rel.tabela_finais({lista}, {rots})\nfinais")
    code("rel.grafico_finais(finais); plt.tight_layout(); plt.show()")
    sem_ens = [(i, r) for i, r in finais if i != "E5_ensemble"]
    code(f"tab = rel.recall_por_classe({json.dumps([i for i, _ in sem_ens])}, {json.dumps([r for _, r in sem_ens])})\n"
         "display(tab)\nrel.heatmap_recall(tab); plt.tight_layout(); plt.show()")
    code("""
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
rel.matriz_confusao('E1_final_s42', 'E1 ConvNeXt 384 px', axs[0])
rel.matriz_confusao('E2_384_final_s42', 'E2 MaxViT 384 px', axs[1])
plt.tight_layout(); plt.show()
""")
    md("""
### Testes (T1 = ISIC 2018, comparação com dermatologistas; T2 = metade do teste ISIC 2019)

*Ainda não abertos.* Serão rodados uma única vez nos finalistas (`python -m dermato.testar`), com
bootstrap pareado contra o E0 (2.000 amostras) e a comparação com os dermatologistas de Tschandl et al.
(2020). Esta seção mostrará a diferença entre validação e teste.

### Conclusão (validação; testes ainda fechados)

- O que mais rendeu: **parâmetros** (+3,9 pontos no B0) e **resolução 384 px** (+10 pontos em melanoma),
  não a troca de arquitetura: ConvNeXt e MaxViT empatam na mesma resolução; a fusão com Transformer (E3) e
  o PanDerm (E4) não somaram.
- Melhor modelo individual: **E1 (ConvNeXt-Tiny 384 px)** — 3 sementes: bal_acc 0,764 ± 0,005, spec95
  0,766 ± 0,006, AUC melanoma 0,934 — empatado em acurácia com E3 e E2 a 384 px, um pouco à frente em spec95.
- Melhor sistema para triagem: **E5 (média dos 4 + temperatura)** — melhor ordenação (AUC maligno 0,950) e
  probabilidades calibradas (ECE 0,056).
- Ceratose actínica continua a classe mais difícil (0,57–0,63) em todas as etapas.
- Candidatos para abrir os testes: **E0 (régua), E1 (3 sementes) e E5**.
""")

    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                                    "language": "python"}})
    nbf.write(nb, SAIDA)
    print(f"-> {SAIDA} ({len(cells)} células)")
    if not a.sem_executar:
        r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                            "--inplace", "--ExecutePreprocessor.timeout=600", str(SAIDA)], cwd=REPO)
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()

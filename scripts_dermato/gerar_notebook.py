"""Gera notebook_dermatoscopia.ipynb (registro das iterações E0–E5) a partir dos resultados salvos.

Uso:
    python scripts_dermato/gerar_notebook.py            # gera e executa (só lê resultados; CPU, ~1 min)
    python scripts_dermato/gerar_notebook.py --sem-executar

Números da divisão v2 (sem imagens repetidas entre splits, 14/11). As buscas E0–E4 rodaram na divisão
v1; as configurações vencedoras foram retreinadas e avaliadas na v2. Os textos resumem o DIARIO.md.
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


def secao_busca(nome, titulo, como_funciona, variado_extra, funcionou, v1=True):
    md(f"## {titulo}\n\n### 1. Como funciona\n\n{como_funciona}")
    aviso = ("\n\n*Busca feita na divisão v1 (validação com imagens repetidas): a tabela abaixo é da v1. A "
             "configuração vencedora foi retreinada na v2; os números do texto são da v2.*") if v1 else ""
    md(f"### 2. O que foi variado\n\n{variado_extra}{aviso}")
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

    md("""
# Classificador de dermatoscopia — registro das iterações (E0 → E5)

**Problema:** apoiar dermatologistas e serviços de teledermatologia a priorizar lesões suspeitas em
imagens de dermatoscopia, reduzindo encaminhamentos e biópsias desnecessárias sem deixar passar casos
de câncer. O resultado é **apoio à priorização, nunca diagnóstico** (o diagnóstico é a histopatologia).

Este notebook **não treina nada**: lê `resultados/experimentos.csv` (uma linha por treino),
`resultados/optuna.db` (todas as tentativas das buscas) e as previsões de validação salvas em
`resultados/previsoes/`. Roda em qualquer PC com o repositório, sem as imagens e sem GPU. O raciocínio
completo, na ordem em que aconteceu, está em `resultados/DIARIO.md`.

> **Divisão v2 (14/11).** A primeira divisão tinha imagens repetidas entre treino, validação e testes
> (seção "Imagens repetidas"). Ela foi refeita e **todos os modelos foram retreinados**; todos os números
> abaixo são da v2. Os resultados antigos estão arquivados em `resultados/v1_com_vazamento/`.
>
> **Testes T1/T2 ainda fechados.** Tudo abaixo é validação. Pelo protocolo, os testes são rodados uma
> única vez, nos finalistas, no fim.
""")
    code("""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image as Img, display
from dermato import relatorio as rel
from dermato.config import CLASSES
rel.estilo()
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
""")

    # ------------------------------------------------------------------ dados e protocolo
    md("""
## Dados e protocolo comum

Divisão fixa (`splits/dermato.csv`, semente 42). T1 = ISIC 2018 (teste da comparação com
dermatologistas); T2 = metade do teste ISIC 2019, sorteada **por grupo**; validação = 15% do conjunto de
treino, sorteada por grupo (paciente, lesão ou imagens repetidas), estratificada por classe.
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
| F1 e F2 por classe, erros tipo I e II | análise de erros (seção própria) |
| ECE (erro de calibração) | se as probabilidades são confiáveis |

Regras seguidas em todas as etapas: mesma divisão; época, parâmetros, tau (ajuste de prior) e limiar
escolhidos **só na validação**; TTA de 8 vistas em todos; busca automática com Optuna (corte das
tentativas ruins a partir da 4ª época); cada rodada registrada em `experimentos.csv`.

**Ruído entre rodadas** (medido com 3 sementes): a mesma configuração varia **~1 ponto de acurácia
balanceada** e **~1–2 pontos de spec95**. Diferenças menores que isso são empate.
""")

    # ------------------------------------------------------------------ duplicatas
    md("""
## Imagens repetidas entre splits (14/11) e divisão v2

A divisão original garantia que nenhum `isic_id` e nenhum grupo (paciente/lesão) caísse em dois splits,
mas só onde a base informa o grupo. O teste do ISIC 2019 **não tem `lesion_id`** e as bases se sobrepõem
(MILK10k e ISIC 2020 reaproveitam imagens do arquivo ISIC), então a mesma lesão aparecia com IDs
diferentes em treino, validação e testes.

**Como as repetidas foram encontradas** (`python -m dermato.duplicatas pares`). Um par é a mesma lesão se:
- o **cosseno entre os vetores do PanDerm** (centralizados pela média de todas as imagens) é ≥ 0,90, **ou**
- há **≥ 25 pontos ORB casados** após RANSAC (homografia; também com a imagem espelhada). Pega cópias
  recoloridas ou recortadas que o PanDerm não pega (cos 0,71–0,88).

Calibração do ORB: duplicatas conhecidas têm mediana de 312 pontos (23 de 26 acima de 25); lesões
diferentes, mediana 0 (1 de 200 acima). Testados e descartados: hash perceptual e correlação de pixels
(nevos lisos diferentes se correlacionam acima de 0,95).

**Quanto vazava na v1** (imagens com par no treino):

| Split | Vazamento v1 |
|---|---|
| T1 (ISIC 2018) | 16% |
| T2 (metade teste ISIC 2019) | 46% — as metades tinham sido sorteadas por imagem e o BCN fotografa a mesma lesão várias vezes |
| Validação | 12% |

**Divisão v2:** os pares viram grupos (componentes conexos + grupos originais) e a divisão por grupo é
refeita; imagens de treino com par no T1 (e as da mesma lesão) foram removidas, para manter o T1 inteiro.
A divisão tem uma asserção: **nenhum par cruza splits**.
""")
    code("""
pares = pd.read_csv('splits/pares_duplicatas.csv.gz')
rem = pd.read_csv('splits/removidas_duplicatas.csv')
print(f"{len(pares)} pares encontrados:", pares['criterio'].value_counts().to_dict())
print(f"{len(rem)} imagens de treino removidas (par com o T1):", rem['motivo'].value_counts().to_dict())
sp_id = sp.set_index('isic_id')['split']
cruza = (pares['a'].map(sp_id) != pares['b'].map(sp_id)) & pares['a'].isin(sp_id.index) & pares['b'].isin(sp_id.index)
print('pares que cruzam splits na v2:', int(cruza.sum()))
dup = pd.read_csv('resultados/duplicatas.csv')
dup.groupby('split')['cos'].agg(imagens='size', cos_max='max').round(4)
""")
    md("""
Pares mais parecidos que **sobraram** entre cada teste e o treino na v2 (à esquerda o teste, à direita o
vizinho do treino). Abaixo de 0,90, e sem casar pelo ORB, são lesões diferentes que só se parecem:
""")
    code("""
for split in ['t2', 't1']:
    for faixa in ['0.85', '0.80']:
        print(f'{split.upper()}, cos a partir de {faixa}')
        display(Img(filename=f'resultados/figuras/duplicatas_{split}_{faixa}.jpg', width=900))
""")
    md("""
**Quanto a v1 inflava a validação.** Tirando da validação v1 as imagens vazadas, os modelos v1 perdiam
1–2,6 pontos (ex.: E1 bal_acc 0,769 → 0,756, spec95 0,773 → 0,760). **A ordem entre os modelos não mudava**:
o vazamento afetava todos de forma parecida. Os números v2 abaixo confirmam: ficaram próximos dessa
estimativa, e as conclusões entre etapas se mantêm.
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
- Acurácia balanceada **0,696**, spec95 **0,659**, AUC melanoma 0,920 (validação v2, TTA 8).
- Melhor época 7 de 8: partindo do ImageNet, 8 épocas são poucas (o Keras original partia de pesos já
  ajustados no ISIC 2019). Primeira hipótese para a busca: mais épocas.
- tau alto (1,45): treinado na distribuição natural (muitos nevos), precisa de desconto forte do prior.
""")

    secao_busca("E0_busca", "E0_busca — a mesma rede com parâmetros otimizados", """
Mesma EfficientNetV2-B0 do E0, agora com todos os parâmetros livres (inclusive resolução 224/288/384).
Separa "ganho de ajustar parâmetros" de "ganho de trocar de rede": a comparação justa de E1/E2 é contra
esta etapa, não só contra a receita.
""", "Todos os parâmetros comuns do plano (faixas abaixo), orçamento de 1,25 h + ~20 min.", """
- **+2,3 pontos de acurácia balanceada (0,696 → 0,719) só com parâmetros**, mesma rede e resolução
  (na v1 o ganho medido era 3,9). spec95 0,659 → 0,670.
- lr ~4× maior (3,8e-4), suavização de rótulos 0,11, dropout/drop path maiores e **todos os nevos**
  (mais dados) explicam o ganho; lr < 3e-5 foi ruim em qualquer combinação.
- Lote 16 nas duas melhores (mais passos por época); augmentation leve e pouco mixup ganharam da forte.
- **Sem Shades of Gray** venceu — surpresa; repetiu-se na E1 e na E2.
""")

    secao_busca("E1", "E1 — CNN mais forte (ConvNeXt-Tiny, até 384 px)", """
ConvNeXt-Tiny (ImageNet-22k) com resolução até 384 px. **Fraqueza atacada:** melanoma reconhecido em só
~72% — detalhes finos (rede de pigmento, bordas) se perdem a 224 px. **Hipótese:** mais resolução + CNN
moderna melhoram AUC e acerto de melanoma. A busca foi semeada com a vencedora da E0_busca a 384 px.
""", "Parâmetros comuns + resolução 224/288/384 px; 3,5 h, 16 tentativas.", """
- **Hipótese confirmada.** Acerto em melanoma 0,72 → **0,79** (+7 pontos sobre a E0_busca, 915 melanomas
  na validação); bal_acc 0,757 e **spec95 0,730** (+6 pontos sobre a E0_busca: ~6% a mais dos benignos
  sairiam da fila prioritária com a mesma sensibilidade); AUC melanoma 0,942.
- **Resolução é o fator mais forte**: na busca, a configuração da melhor tentativa a 224 px foi cortada
  na 4ª época, muito atrás da mesma a 384 px.
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
- bal_acc 0,740, spec95 0,660: acima do B0 otimizado (0,719) na mesma resolução, abaixo da E1 a 384 px.
- **Hipótese não confirmada**: a confusão entre ceratoses e carcinoma espinocelular não diminuiu; melhora
  em carcinoma espinocelular (0,72), piora em dermatofibroma.
- O decaimento por camada não mudou nada mensurável na busca.
- Incidente: a VRAM transbordou para a RAM do sistema (épocas 10× mais lentas); corrigido com passos
  menores e teto de memória (ver diário).
""")

    if "E1r_final_s42" in tem:
        md("""
## E1r — rodada extra da E1 (refino)

Busca estreitada em volta do que funcionou (sem lr baixo, sem augmentation forte, sem SoG), testando
duas hipóteses diretas: **448 px** (mais resolução ainda ajuda?) e **gama da focal acima de 3**.

*Busca feita na divisão v1 (tabela abaixo); retreino na v2.*
""")
        code("faixas, melhores, vencedora = rel.resumo_busca('E1r')\ndisplay(melhores); print(vencedora)")
        md("""
- **448 px piorou** na busca e é ~40% mais lento: a família ConvNeXt fica em 384 px.
- Gama 4,5: na v2, bal_acc 0,760 × 0,754 da E1 (média de 3 sementes) — empate —, mas **spec95 pior**
  (0,708 × 0,723) e ECE ainda mais alto (0,39). A E1 continua a referência.
- Foi daqui que veio a medida de **ruído ≈ 1 ponto** usada em todo o notebook (a busca deu 0,775 e o
  retreino 0,765 só por mudar a sequência de augmentation).
""")

    secao_busca("E2_384", "E2 a 384 px — arquitetura × resolução", """
A mesma MaxViT na resolução da E1, para separar "a MaxViT é pior" de "224 px é pior". Busca estreitada
e semeada com a vencedora da E2.
""", "Espaço estreitado (faixas abaixo), 5 h, ~1 h por tentativa.", """
- **Resolução ajuda a MaxViT em acurácia**: 224 → 384 px, bal_acc 0,740 → 0,756 (média de 3 sementes).
- **Em acurácia balanceada, as arquiteturas empatam** (0,756 ± 0,012 × 0,754 ± 0,004 da E1), mas **na
  triagem a MaxViT fica atrás**: spec95 0,667 ± 0,013 × 0,723 ± 0,008 — as 3 sementes da MaxViT abaixo de
  todas as do ConvNeXt. Na v1 essa diferença não aparecia; a validação com repetidas escondia.
- A MaxViT é melhor em carcinoma espinocelular (0,71 × 0,63), o ConvNeXt em ceratose actínica e na
  ordenação maligno × benigno: erram em lesões diferentes (bom para a E5).
""")

    if "PanDerm_rapido" in set(exp["etapa"]):
        md("""
## Teste rápido do PanDerm (antes da E4)

Vetores congelados do PanDerm (ViT-L/16, modelo de fundação de dermatologia), sem ajuste fino, com três
classificadores simples na CPU.
""")
        code("""
pr = rel.experimentos().query("etapa == 'PanDerm_rapido'")
pr[['familia','val_bal_acc','val_spec95','val_mel_auc','val_mal_auc','tempo_treino_min']].round(3)
""")

    secao_busca("E3", "E3 — fusão de dois ramos (ConvNeXt + Swin-Tiny)", """
ConvNeXt (detalhe local, partindo dos pesos da E1) e Swin-Tiny (relações entre regiões distantes) em
paralelo, mais metadados; a junção (concatenação, porta aprendida ou atenção cruzada) decide quanto
confiar em cada um. **Hipótese:** dois olhares separados se complementam mais do que misturados numa
rede (E2).
""", "Parâmetros comuns + tipo de junção, lr do ramo Swin (0,1–1× o do ConvNeXt), épocas com os ramos "
         "congelados (0–3), cabeça 256/512; 3 h.", """
- **A fusão não superou a E1 sozinha**: 3 sementes, bal_acc 0,752 ± 0,004 × 0,754 ± 0,004; spec95
  0,716 ± 0,010 × 0,723 ± 0,008 (empate). Hipótese não confirmada nesta forma: o Swin a 224 px não
  acrescentou informação que o ConvNeXt a 384 px já não tivesse.
- Como o ramo ConvNeXt já parte treinado (pesos da E1 v2), **lr baixo** funcionou melhor e a melhor época
  chega cedo (1–3).
- Melhor calibrada que a E1 (ECE 0,19–0,22 × 0,31).
""")

    secao_busca("E4", "E4 — fusão + conhecimento de dermatologia (PanDerm)", """
Terceiro ramo na fusão da E3: o vetor congelado do PanDerm (1.024 números por imagem, sem ajuste fino).
**Fraqueza atacada:** classes raras com poucos exemplos (dermatofibroma, lesão vascular, carcinoma
espinocelular). **Hipótese:** um modelo que viu muito mais pele compensa a falta de exemplos raros. A
busca foi semeada com a vencedora da E3 + PanDerm.
""", "Os mesmos parâmetros da E3 + dropout no vetor do PanDerm (0–0,6) e vetor normalizado ou não; 3 h.", """
- v2: bal_acc 0,760, spec95 0,715 (uma semente) × E3 0,752 / 0,716 (3 sementes): **dentro do ruído**.
- **Hipótese não confirmada**: nas classes raras não há ganho consistente (carcinoma espinocelular 0,61,
  dermatofibroma 0,82, iguais à E1). Os vetores congelados repetem informação que a CNN treinada já tem.
""")

    if "E5_ensemble" in tem:
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
ens[['membros','metodo','bal_acc','spec95','mel_auc','mal_auc','ece','temperatura']].head(8).round(4)
""")
        md("""
### 3–4. Resultado e leitura (metade B da validação)
- Vencedor na v2: **média simples de E2 a 384 px + E4 + temperatura**: bal_acc **0,777**, spec95 0,700,
  AUC melanoma **0,947**, AUC maligno **0,945**, ECE **0,052**.
- Ganhos sobre os modelos sozinhos: **acurácia (+2 pontos sobre a E1), ordenação de melanoma e
  calibração** (ECE 0,31 → 0,05). Em spec95 não ganha da E1 sozinha (0,700 × 0,723): a MaxViT, fraca em
  triagem, puxa para baixo.
- Média simples ganhou da ponderada e do LightGBM: com metade da validação para ajustar, mais parâmetros
  não compensam.
""")

    sementes = [f for f in ("E1", "E2_384", "E3") if f"{f}_final_s44" in tem]
    rot_s = {"E1": "E1 ConvNeXt 384", "E2_384": "E2 MaxViT 384", "E3": "E3 fusão ConvNeXt+Swin"}
    if sementes:
        md("""
## Finalistas: 3 sementes cada

O ruído entre rodadas (~1 ponto) é do tamanho das diferenças entre os melhores modelos. Os 3 maiores
bal_acc foram retreinados com as sementes 43 e 44 (mesma configuração), como manda o protocolo: **uma
diferença menor que a variação entre sementes é empate.**
""")
        code(f"display(rel.tabela_sementes({json.dumps(sementes)}, {json.dumps([rot_s[f] for f in sementes])}))\n"
             f"rel.grafico_sementes({json.dumps(sementes)}, {json.dumps([rot_s[f] for f in sementes])}); "
             "plt.tight_layout(); plt.show()")
        md("""
- **Acurácia balanceada: empate.** E2 a 384 px 0,756 ± 0,012, E1 0,754 ± 0,004, E3 0,752 ± 0,004.
- **Especificidade a 95%: a E1 lidera** (0,723 ± 0,008), empatada com a E3 (0,716 ± 0,010) e bem à frente
  da E2 a 384 px (0,667 ± 0,013).
- AUC maligno: E1 0,943, E3 0,940, E2 a 384 px 0,935.
- **Melhor modelo individual: E1.** Empata em acurácia, ganha em triagem e AUC, e é o mais rápido
  (30 min de treino × 52 da MaxViT a 384 px). Foi o escolhido para a otimização.
""")

    # ------------------------------------------------------------------ otimização
    if "OT_v2_final_s44" in tem:
        secao_busca("OT_v2", "Otimização do melhor modelo (E1) na v2", """
Nova busca Optuna **em volta da configuração da E1**, já na divisão limpa. Arquitetura, resolução (384 px)
e lote ficam fixos; varia o resto, em faixas centradas na vencedora. **A primeira tentativa é a própria
E1**, então o ganho é medido dentro da mesma busca. A vencedora foi retreinada com 3 sementes.
""", "lr (½× a 2× o da E1, log), weight decay, épocas (8–16), aquecimento, gama da focal, suavização, "
            "augmentation, mixup/cutmix, dropout, drop path, **decaimento de lr por camada** (ainda não testado "
            "no ConvNeXt), limite de nevos e épocas só da cabeça. 4,8 h, 17 tentativas (13 cortadas).", """
- Na busca, a tentativa 2 deu **0,773** contra 0,757 da E1 (tentativa 0): +1,6 ponto. Mudanças: cutmix no
  lugar de mixup, augmentation média, gama 2 (× 3), suavização 0,06 (× 0,16), decaimento por camada 0,86.
- **Com 3 sementes, o ganho quase some**: 0,759 ± 0,012 × 0,754 ± 0,004 — empate. As sementes 43 e 44
  deram 0,752: o 0,773 foi uma semente de sorte, escolhida justamente por ser a maior (*maldição do
  vencedor*).
- **Na triagem, piorou**: spec95 0,685 ± 0,011 × 0,723 ± 0,008 (as 3 sementes abaixo de todas as da E1);
  AUC maligno 0,939 × 0,943. Ganho real só em calibração (ECE 0,31 → 0,16) e em carcinoma espinocelular.
- **Conclusão:** a configuração da E1 já estava perto do ótimo para esta rede e estes dados; mais busca de
  parâmetros não melhora o modelo individual. O ganho que sobra vem de combinar modelos (E5_OT).
""", v1=False)
        code("display(rel.tabela_sementes(['E1', 'OT_v2'], ['E1 (base)', 'OT_v2 (otimizado)']))\n"
             "rel.grafico_sementes(['E1', 'OT_v2'], ['E1 (base)', 'OT_v2 (otimizado)']); plt.tight_layout(); plt.show()")
        code("""
cfg = rel.experimentos().set_index('id')['config']
b, o = json.loads(cfg['E1_final_s42']), json.loads(cfg['OT_v2_final_s42'])
pd.DataFrame({'E1': {k: b.get(k) for k in o if b.get(k) != o[k] and k != 'etapa'},
              'OT_v2': {k: o[k] for k in o if b.get(k) != o[k] and k != 'etapa'}})
""")
    if "E5_OT" in tem:
        md("""
### E5_OT — ensemble com o modelo otimizado

O mesmo procedimento da E5 (ajuste na metade A, avaliação na metade B), com o OT_v2 no lugar da E1.
""")
        code("""
eot = pd.read_csv('resultados/ensemble_busca_E5_OT.csv')
eot[['membros','metodo','bal_acc','spec95','mel_auc','mal_auc','ece','temperatura']].head(6).round(4)
""")
        md("""
- Vencedor: **média de E2 a 384 px + E4 + OT_v2**: bal_acc **0,782**, spec95 **0,723**, AUC melanoma
  0,949, AUC maligno 0,947, ECE **0,041** — o melhor em todas as métricas.
- Sobre a E5: +0,5 ponto em acurácia (empate) e **+2,4 em spec95** (0,700 → 0,723). O OT_v2, melhor em
  carcinoma espinocelular, complementa a MaxViT.
- Cautela: o membro OT_v2 é a semente 42, a escolhida na validação (otimista), e a metade B tem ~2.600
  imagens (ruído de spec95 ~2 pontos). O teste dirá se o ganho sobre a E5 é real.
""")

    # ------------------------------------------------------------------ erros
    if (REPO / "resultados" / "erros_classes.csv").exists():
        md("""
## Análise de erros: F1/F2 por classe e erros tipo I/II

`python -m dermato.erros` lê as previsões de validação e calcula:
- **por classe**: precisão, sensibilidade, **F1** (média harmônica das duas) e **F2** (pesa a sensibilidade
  4× mais: perder um câncer é pior que um alarme falso);
- **maligno × benigno** (malignos = ceratose actínica, carcinoma basocelular, melanoma, carcinoma
  espinocelular) e **melanoma × resto**, com **erro tipo I** (falso positivo: benigno marcado como
  maligno = FP/(FP+VN)) e **erro tipo II** (falso negativo: maligno que passou = FN/(FN+VP)), em dois pontos:
  o **limiar de triagem** (95% de sensibilidade, o uso pretendido) e a **classe prevista**.

Modelos de 3 sementes aparecem como média. Atenção: nos ensembles, a temperatura e o limiar foram ajustados
na metade A, mas a tabela usa a validação inteira, então eles ficam um pouco otimistas aqui.
""")
        code("""
ec = pd.read_csv('resultados/erros_classes.csv')
ec['modelo'] = ec['modelo'].str.replace(r'_s4[234]$', '', regex=True)
ordem = ['E0_final', 'E1_final', 'OT_v2_final', 'E5_ensemble', 'E5_OT']
f = ec.groupby(['classe', 'modelo'])[['F1', 'F2', 'precisao', 'sensibilidade']].mean()
nomes = {c: rel.NOMES.get(c, c) for c in ec['classe'].unique()}
for m in ['F1', 'F2']:
    t = f[m].unstack()[ordem].rename(index=nomes)
    print(m); display(t.round(3))
""")
        code("""
fig, ax = plt.subplots(figsize=(10, 3.6))
t = f['F2'].unstack()[ordem].drop('macro').rename(index=nomes)
t.plot.bar(ax=ax, width=0.8); ax.set_ylabel('F2'); ax.set_xlabel(''); ax.set_ylim(0, 1)
ax.legend(ncol=5, fontsize=8, loc='lower right'); plt.xticks(rotation=0); plt.tight_layout(); plt.show()
""")
        md("""
**F1/F2 por classe — leitura:**
- **F2 macro**: E0 0,620 → E1 0,702 → OT_v2 0,721 → E5 0,734 → **E5_OT 0,748**. Cada passo ganha nas
  classes difíceis; a otimização ganhou em F1/F2 (precisão de lesão vascular e dermatofibroma), mesmo sem
  ganhar em spec95.
- **Classes difíceis**: ceratose actínica (F1 ~0,48–0,51), **dermatofibroma** (F1 0,40–0,56: sensibilidade
  boa, 0,82, mas **precisão baixa**, 0,27–0,43 — com só 45 casos, poucos falsos positivos derrubam a
  precisão) e carcinoma espinocelular (0,56–0,61).
- **Classes fáceis**: nevo (0,86–0,88), carcinoma basocelular (~0,84) e lesão vascular (0,75–0,89).
- **Melanoma**: F2 ~0,75–0,76, com sensibilidade de 78–79% em todos os modelos fortes: a otimização e o
  ensemble quase não mudam o melanoma.
""")
        code("""
eb = pd.read_csv('resultados/erros_binario.csv')
eb['modelo'] = eb['modelo'].str.replace(r'_s4[234]$', '', regex=True)
cols = ['VP', 'FP', 'FN', 'VN', 'erro_tipo_I', 'erro_tipo_II', 'precisao', 'F1', 'F2']
t = eb.groupby(['tarefa', 'ponto', 'modelo'])[cols].mean()
t = t.reindex(ordem, level='modelo')
t.round(3)
""")
        md("""
**Erros tipo I e II — leitura:**

| | Erro tipo I (benigno → maligno) | Erro tipo II (maligno que passa) |
|---|---|---|
| Triagem (95% sens.), E0 | 34,1% | 5,0% |
| Triagem, E1 | **27,7%** | 5,0% |
| Triagem, OT_v2 | 31,5% | 5,0% |
| Triagem, E5 / E5_OT | 27,2% / 27,7% | 5,4% / 5,0% |
| Classe prevista maligna, E1 | 16,7% | 11,1% |
| Classe prevista maligna, E5 / E5_OT | 17,9% / 16,6% | **9,5%** / 10,2% |
| Melanoma (classe prevista), E1 | 8,2% | 22,0% |
| Melanoma (classe prevista), E5_OT | 8,2% | 20,9% |

- **Na triagem, o erro tipo II é fixado em ~5%** (o limiar é escolhido para isso); o que muda entre
  modelos é o tipo I. A E1 manda 27,7% dos benignos (~850 de 3.081) para a fila prioritária, contra 34,1%
  do E0: ~200 encaminhamentos a menos com a mesma segurança. O OT_v2 piora isso (31,5%).
- **Pela classe prevista**, os modelos erram mais malignos (9,5–12%): a decisão em 8 classes não é feita
  para triagem; para triagem vale o limiar.
- **Melanoma** é o ponto fraco: pela classe prevista, ~1 em cada 5 melanomas é chamado de outra coisa
  (em ~60% dos casos, nevo). Na triagem ele é pego, porque o escore maligno soma as 4
  classes malignas.
""")

    # ------------------------------------------------------------------ linha de raciocínio
    finais = [("E0_final_s42", "E0 receita (B0 224)"), ("E0_busca_final_s42", "E0_busca (B0 224)"),
              ("E2_final_s42", "E2 MaxViT 224"), ("E2_384_final_s42", "E2 MaxViT 384"),
              ("E1_final_s42", "E1 ConvNeXt 384"), ("E1r_final_s42", "E1r ConvNeXt 384 (refino)"),
              ("E3_final_s42", "E3 fusão ConvNeXt+Swin"), ("E4_final_s42", "E4 fusão + PanDerm"),
              ("OT_v2_final_s42", "OT_v2 ConvNeXt otimizado"),
              ("E5_ensemble", "E5 ensemble (metade B)"), ("E5_OT", "E5_OT ensemble (metade B)")]
    finais = [(i, r) for i, r in finais if i in tem]
    md("""
## Linha de raciocínio

0. **Divisão v1 → v2:** imagens repetidas entre splits (T2 46%, T1 16%, validação 12%) inflavam a
   validação em 1–2,6 pontos. Tudo foi refeito sem pares entre splits; a ordem entre modelos não mudou.
1. **E0 → E0_busca:** a receita estava sub-treinada; otimizar parâmetros na mesma rede deu +2,3 pontos.
   Isso virou a régua justa para as redes novas.
2. **E0_busca → E1:** a fraqueza medida era melanoma (~72%). ConvNeXt a 384 px levou melanoma a 79% e
   spec95 a 0,730 — maior ganho do projeto. A resolução, mais que a rede, explicou o salto.
3. **E1 → E2:** a MaxViT não reduziu a confusão entre ceratoses e carcinoma espinocelular. A 384 px
   empata com a E1 em acurácia, mas perde 5,6 pontos de spec95.
4. **E1/E2 → E3:** juntar CNN e Transformer em ramos separados empatou com a CNN sozinha.
5. **E3 → E4:** o PanDerm congelado como terceiro ramo não acrescentou nada consistente.
6. **E1–E4 → E5:** a média de modelos que erram em lesões diferentes, com calibração, deu a melhor
   acurácia (0,777) e calibração (ECE 0,05).
7. **Finalistas (3 sementes):** E1, E3 e E2 a 384 px empatam em acurácia; a E1 lidera em triagem.
8. **Otimização da E1:** o ganho da busca (+1,6) era sorte de semente; com 3 sementes, empate em acurácia
   e perda em triagem. Trocar a E1 pelo otimizado no ensemble (E5_OT) deu o melhor sistema na validação.

A tabela abaixo usa uma semente por modelo (a 42); as linhas dos ensembles são da metade B da validação.
""")
    lista = json.dumps([i for i, _ in finais])
    rots = json.dumps([r for _, r in finais])
    code(f"finais = rel.tabela_finais({lista}, {rots})\nfinais")
    code("rel.grafico_finais(finais); plt.tight_layout(); plt.show()")
    sem_ens = [(i, r) for i, r in finais if not i.startswith("E5")]
    code(f"tab = rel.recall_por_classe({json.dumps([i for i, _ in sem_ens])}, {json.dumps([r for _, r in sem_ens])})\n"
         "display(tab)\nrel.heatmap_recall(tab); plt.tight_layout(); plt.show()")
    code("""
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
rel.matriz_confusao('E1_final_s42', 'E1 ConvNeXt 384 px', axs[0])
rel.matriz_confusao('E5_OT', 'E5_OT ensemble', axs[1])
plt.tight_layout(); plt.show()
""")
    md("""
### Testes (T1 = ISIC 2018, comparação com dermatologistas; T2 = metade do teste ISIC 2019)

*Ainda não abertos.* Na v2 eles não têm nenhuma imagem com par no treino (PanDerm ≥ 0,90 ou ORB ≥ 25).
Serão rodados uma única vez nos finalistas (`python -m dermato.testar`), com bootstrap pareado contra o E0
(2.000 amostras) e a comparação com os dermatologistas de Tschandl et al. (2020).

### Conclusão (validação v2; testes ainda fechados)

- O que mais rendeu: **resolução 384 px** (+7 pontos em melanoma) e **parâmetros** (+2,3 no B0), não a
  troca de arquitetura: ConvNeXt e MaxViT empatam em acurácia na mesma resolução; a fusão com Transformer
  (E3), o PanDerm (E4) e a nova otimização (OT_v2) não somaram ao modelo individual.
- Melhor modelo individual: **E1 (ConvNeXt-Tiny 384 px)** — 3 sementes: bal_acc 0,754 ± 0,004, spec95
  0,723 ± 0,008, AUC maligno 0,943; na triagem, erro tipo I de 27,7% com tipo II de 5%.
- Melhor sistema: **E5_OT (média de E2 a 384 px + E4 + OT_v2, com temperatura)** — bal_acc 0,782, spec95
  0,723, AUC maligno 0,947, ECE 0,041, F2 macro 0,748.
- Ceratose actínica, dermatofibroma (precisão) e carcinoma espinocelular são as classes mais difíceis;
  ~1 em 5 melanomas é classificado como outra lesão (mas pego pela triagem).
- Candidatos para abrir os testes: **E0 (régua), E1 (3 sementes), E5 e E5_OT**.
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

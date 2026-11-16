# Projeto Final – InsperAI (Felipe Gava, Lucas Briones)

Projeto acadêmico de apoio à triagem de lesões de pele: o resultado é sempre apoio à priorização (quem
precisa de biópsia/encaminhamento primeiro), nunca um diagnóstico. Relatório da entrega intermediária
em `relatorios/relatorio_intermediaria.pdf`.

## Mapa do repositório

O projeto passou por fases diferentes. **Comece pelo `notebook_dermatoscopia.ipynb`**, que é o trabalho atual.

| Fase | Notebook | Código e resultados |
|---|---|---|
| **Atual**: classificador de dermatoscopia, E0 → E5 e otimização (PyTorch) | `notebook_dermatoscopia.ipynb` (gerado por `scripts_dermato/gerar_notebook.py`; não editar à mão) | `dermato/`, `scripts_dermato/`, `resultados/` (inclui o `DIARIO.md`), `splits/` |
| v1: um único modelo para foto de celular e dermatoscopia (TensorFlow) | `notebook_v1_celular.ipynb` | `pipeline.py`, `prepare_data.py`, `scripts/`, `modelos/pintas/`, `data/processed/` |
| Triagem de erupções, alergias e queimaduras (**pausada**) | `triagem_alergias/triagem_alergias.ipynb` | `triagem_alergias/` |
| Treino no Kaggle (**arquivado, nunca executado**) | `arquivo/kaggle/kaggle_professor.ipynb` | `arquivo/kaggle/` (ver o README da pasta) |

## Classificador de dermatoscopia (PyTorch) — escopo atual

O projeto passou a focar só em **dermatoscopia**. O código está em `dermato/`; as seções a partir de
"Modelo v1" descrevem a fase anterior (TensorFlow, foto de celular).

**Resultados na validação** (divisão v2, sem imagens repetidas entre splits; 5.319 imagens, TTA 8,
semente 42; os testes T1/T2 ainda não foram abertos):

| Modelo | Resolução | Acurácia balanceada | Especificidade a 95% de sensibilidade | AUC melanoma |
|---|---|---|---|---|
| E0 — receita do modelo v1 (EfficientNetV2-B0) | 224 | 0,696 | 0,659 | 0,920 |
| E0_busca — B0 com parâmetros otimizados | 224 | 0,719 | 0,670 | 0,926 |
| E1 — ConvNeXt-Tiny | 384 | 0,757 | 0,730 | 0,942 |
| OT_v2 — E1 otimizada | 384 | 0,773 | 0,682 | 0,933 |
| E5 — ensemble (E2 a 384 px + E4) + temperatura\* | — | 0,777 | 0,700 | 0,947 |
| **E5_OT — ensemble (E2 a 384 px + E4 + OT_v2) + temperatura\*** | — | **0,782** | **0,723** | **0,949** |

\* Ensembles ajustados na metade A da validação e avaliados na metade B. As médias de 3 sementes dos
finalistas e a análise de erros (F1/F2, tipo I/II) estão no notebook. Os números da divisão v1, que
tinha imagens repetidas entre splits e inflava a validação em 1–2,6 pontos, estão arquivados em
`resultados/v1_com_vazamento/`.

Raciocínio, mudanças e números de cada etapa: `resultados/DIARIO.md`; uma linha por treino em
`resultados/experimentos.csv`; todas as tentativas das buscas em `resultados/optuna.db`.

### Pesos dos modelos e Git LFS

Os arquivos de pesos (`resultados/modelos/*.pt`, de 24 a 232 MB) são grandes demais para o git
comum (o GitHub recusa arquivos acima de 100 MB). Por isso eles ficam no **Git LFS** (Large File
Storage): o repositório guarda só um "ponteiro" de poucos bytes e o arquivo de verdade fica no
armazenamento LFS do GitHub, baixado junto com o clone.

| Arquivo | Tamanho | Modelo |
|---|---|---|
| `E0_final_s42.pt` | 24 MB | E0, receita do modelo v1 (EfficientNetV2-B0, 224 px) |
| `E0_busca_final_s42.pt` | 24 MB | E0 com parâmetros otimizados |
| `E1_final_s42.pt`, `_s43.pt`, `_s44.pt` | 111 MB cada | E1, ConvNeXt-Tiny 384 px (3 sementes; a s42 é o padrão do `prever.py`) |
| `E2_384_final_s42.pt` | 122 MB | E2, MaxViT-Tiny 384 px (membro dos ensembles) |
| `E4_final_s42.pt` | 232 MB | E4, fusão ConvNeXt + Swin + PanDerm (membro dos ensembles) |
| `OT_v2_final_s42.pt` | 111 MB | OT_v2, E1 otimizada (membro do E5_OT) |

Os pesos da divisão v1 (E0, E0_busca e E1) ficam em `resultados/v1_com_vazamento/modelos/`, também no LFS.

**Antes de clonar (uma vez por computador):**
```
git lfs install
```
O Git LFS já vem com o Git para Windows; no Linux/macOS, instale o pacote `git-lfs` antes.

**Clonar o repositório (já vem com os pesos):**
```
git clone https://github.com/felipegava1/ProjetoFinal-InsperAI-FelipeGava-LucasBriones
```

**Se o clone já foi feito sem o LFS**, os `.pt` vêm como arquivos de texto de ~130 bytes (os
ponteiros) e o `prever.py` falha ao carregar. Para baixar os arquivos de verdade:
```
git lfs install
git lfs pull
```

**Conferir se os pesos estão íntegros:** `git lfs ls-files` lista os arquivos (8 da v2 e 3 da v1); os
códigos devem bater com `resultados/modelos_sha256.txt`, que também traz os de pesos não versionados (no Windows: `certutil -hashfile resultados\modelos\E1_final_s42.pt SHA256`).

**Cota:** o armazenamento e o download de arquivos LFS contam na cota do dono do repositório no
GitHub (Settings → Billing). Os 11 pesos somam ~1 GB, e cada clone completo baixa tudo isso.
Para clonar sem baixar os pesos: `GIT_LFS_SKIP_SMUDGE=1 git clone ...`
(no PowerShell: `$env:GIT_LFS_SKIP_SMUDGE=1` antes do clone).

**Adicionar um modelo novo**: copie o `.pt` para `resultados/modelos/`
e faça `git add` normalmente — o `.gitattributes` já manda todo `resultados/modelos/*.pt` para o LFS.

### Usar um modelo treinado (sem as bases de dados)

1. Dependências (Python 3.12):
   ```
   pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
   pip install -r requirements.txt
   ```
   Sem GPU: `pip install torch torchvision` e use `--cpu` abaixo.
2. Confira que os pesos vieram de verdade em `resultados/modelos/` (seção **Pesos dos modelos e
   Git LFS** acima). As calibrações (tau e limiar de triagem) estão em `resultados/previsoes/`.
3. Rode:
   ```
   python -m dermato.prever caminho/para/imagens --modelo E1_final_s42 [--meta metadados.csv] [--cpu]
   ```
   Sai um `previsoes.csv` com a classe, as probabilidades das 8 classes, o escore de malignidade e a
   triagem (`prioritaria` se o escore passa o limiar de 95% de sensibilidade da validação).
   `--meta` (colunas `arquivo, idade, sexo, local`) é opcional, mas recomendado: os metadados mudam
   as probabilidades em até ~0,06.

### Reproduzir o treino

```
python scripts_dermato/baixar_dados.py        # ISIC 2018/2019, MILK10k, dermatologistas (~15 GB)
python scripts_dermato/baixar_isic2020.py     # só as 5.993 imagens do ISIC 2020 usadas
python -m dermato.preparar cache              # cache de 448 px (a divisão fixa já está em splits/)
python -m dermato.fila --ate E2 --horas "E0_busca=1.25,E1=3.5,E2=3.5"
```
Os dados vão para `DERMATO_DATA` (padrão `F:\dermato\data`) e os resultados para
`DERMATO_RESULTADOS`. Etapas: E0 receita → E0_busca → E1 (ConvNeXt) → E2 (MaxViT) → E3 (fusão
CNN + Swin) → E4 (+ PanDerm) → E5 (ensemble). Os testes T1/T2 só são abertos com
`python -m dermato.testar`, nos finalistas.

## Modelo v1 (TensorFlow, foto de celular) — fase anterior

O trabalho desta fase (código e texto) está no `notebook_v1_celular.ipynb`.

Um **único modelo de pintas** (EfficientNetV2-B0, 224 px, com idade, sexo e região do corpo) atende
foto de celular e dermatoscopia. A ideia central: ensinar a rede a reconhecer lesões em foto de celular
**sem usar nenhuma foto de celular no treino**, com fotos clínicas comuns e uma augmentation que simula
celular (`pipeline.simular_celular`).

| Teste | Acurácia | Acurácia balanceada | AUC melanoma | AUC maligno × benigno |
|---|---|---|---|---|
| Dermatoscopia (ISIC 2018, 1.511 imagens) | 82,8% | 75,0% | 0,931 | 0,945 |
| Foto de celular (PAD-UFES-20 + HIBA, 2.643 fotos) | 38,9% | 47,8% | 0,871 | 0,868 |

A triagem de erupções, alergias e queimaduras está **pausada** e separada em `triagem_alergias/`.

### Preparando a base de dados

As imagens (~17 GB) ficam só no seu PC, fora do repositório e fora do OneDrive.

1. Baixe as duas fontes:
   - **HAM10000**: `dataverse_files.zip` em
     https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
     (*Access Dataset* → *Download ZIP*).
   - **ISIC 2019**: em https://challenge.isic-archive.com/data/#2019, os arquivos
     `ISIC_2019_Training_Input.zip`, `ISIC_2019_Training_GroundTruth.csv`,
     `ISIC_2019_Training_Metadata.csv`, `ISIC_2019_Test_Input.zip` e
     `ISIC_2019_Test_Metadata.csv`, todos na mesma pasta (os zips podem estar extraídos ou não).
2. Instale as dependências: `pip install pandas tensorflow pillow`
3. Na pasta do repositório, rode:
   ```
   python prepare_data.py CAMINHO/dataverse_files.zip PASTA_DO_ISIC2019
   ```
   O primeiro argumento também aceita a pasta onde o `dataverse_files.zip` foi extraído.
   A base vai para `~/datasets/lesoes_pele` (no Windows, `C:\Users\<usuario>\datasets\lesoes_pele`).
   Para outro lugar, passe o destino como terceiro argumento e defina a variável de ambiente
   `LESOES_DIR` com esse caminho, para que o notebook a encontre.
4. Depois de pronto, os arquivos baixados podem ser apagados.

#### Estrutura gerada

```
lesoes_pele/
├── train/<dx>/   21486 imagens  ISIC 2019, ~85% das lesões
├── val/<dx>/      3845 imagens  ISIC 2019, ~15% das lesões
├── test/          1511 imagens  ISIC 2018 Task 3, rótulo no labels.csv
├── unlabeled/     6726 imagens  resto do teste do ISIC 2019, sem rótulo
├── masks/                       máscaras de segmentação (só imagens do HAM10000)
└── labels.csv                   uma linha por imagem
```

- **Classes (`dx`)**: `ak`, `bcc`, `bkl`, `df`, `mel`, `nv`, `scc`, `vasc`.
- **Teste**: o ISIC 2018 só tem as 7 classes do HAM10000, em que `akiec` junta `ak` e `scc`. Por
  isso o rótulo do teste está em `dx_ham`; o treino também tem `dx_ham` para comparação.
- **Validação**: sorteada por `lesion_id` com semente fixa, então fotos da mesma lesão nunca
  ficam em train e val ao mesmo tempo, e todos que rodarem o script recebem a mesma divisão.
- **`labels.csv`** traz também `source` (HAM, BCN, MSK), `malignant`, idade, sexo, região do corpo
  (`site`, nas 8 regiões do ISIC 2019) e, só para o HAM, `dx_type` e `dataset`.
- A imagem `ISIC_0035068` está no gabarito do ISIC 2018 mas não vem no zip, por isso é descartada.

#### Usando no notebook

```python
from prepare_data import load_labels, CLASSES
df = load_labels()  # DataFrame com caminhos absolutos
```

O `notebook_v1_celular.ipynb` já carrega a base e monta os datasets do TensorFlow na primeira seção.
O carregamento em `tf.data` fica em `pipeline.py`.

### Modelo

O modelo do projeto está em `modelos/pintas/` (receita, resultados e limitações no `README.md` da pasta
e nas seções 2 a 4 do notebook):

- `melhor.keras`: o modelo (`keras.models.load_model("modelos/pintas/melhor.keras", compile=False)`).
  Entradas: `image` (224×224×3, 0–255, com Shades of Gray, ver `pipeline.py`) e `meta` (14 valores,
  `pipeline.meta_features`). Saída: logits das 8 classes.
- `pesos_de_partida.keras`: os pesos só do ISIC 2019, de onde o ajuste começa; servem apenas para retreinar.
- `probs_isic2018.npz`, `probs_val_dermatoscopia.npz`, `previsoes_celular.csv`: previsões salvas.

O notebook não treina o modelo: ele carrega esses arquivos e reavalia tudo, então roda sem os 54 GB da
base ampliada. Para a exploração, e a demonstração da simulação, ele precisa da base deste
guia. Retreinar exige a base ampliada (seção seguinte) e leva cerca de 2,5 h na CPU:

```
uv run scripts/treinar_simulando_celular.py --conjunto dermatoscopia+fotos
```

### Bases de dados ampliadas (fotos de celular)

O uso final é em foto de celular, então além do ISIC 2019/2018 baixamos as outras bases públicas
com rótulo, inclusive as de foto comum. As imagens (~54 GB brutos, 483 mil arquivos) ficam **fora
do repositório**, em `data/` (gitignorado); o que vai para o git é o que reproduz tudo: os scripts
de download e preparo em `scripts/`, o manifesto comprimido (`data/processed/manifesto.csv.gz`) e
os resumos (`data/processed/resumo_por_classe.csv`, `verificacao.csv`, `qualidade/`).

| Base | O que é | Imagens | Melanomas | Aparelho | Como obter |
|---|---|---|---|---|---|
| ISIC 2019 (treino + teste rotulado) | HAM10000 + BCN_20000 + MSK, 8 classes | 33.569 | 5.849 | dermatoscópio | `baixar_dados.py isic2019` |
| ISIC 2020 | SIIM-ISIC, só as com diagnóstico específico entram nas 8 classes | 33.126 | 584 | dermatoscópio | `baixar_dados.py isic2020` |
| ISIC 2024 (SLICE-3D) | recortes de fotografia corporal total 3D, 393 malignos; rótulo só em conjunto (benigno/maligno/indeterminado) | 401.059 | 157 | TBP (Vectra) | `baixar_dados.py isic2024` |
| MILK10k | 5.240 lesões, cada uma com dermatoscopia **e** foto de perto, 96% com histopatologia | 10.480 | 900 | dermatoscópio + câmera | `baixar_dados.py milk10k` |
| Fotos clínicas do acervo ISIC | PAD-UFES-20 (2.298, **celular**, Brasil), HIBA (355, **celular**, Argentina), MSKCC (1.276, câmera, corpo inteiro), anônima (100) | 4.029 | 175 | celular / câmera | `baixar_metadados_clinicas.py` e depois `baixar_imagens_clinicas.py` |
| MED-NODE | 70 melanomas e 100 nevos em foto clínica (Groningen) | 170 | 70 | câmera | manual: [cs.rug.nl/~imaging/databases/melanoma_naevi](https://www.cs.rug.nl/~imaging/databases/melanoma_naevi/) → `data/raw/mednode/complete_mednode_dataset.zip` |
| SCIN (Google) | fotos de celular de condições de pele em geral (quase tudo inflamatório/infeccioso), rótulo fraco de dermatologistas | 10.379 | 3 | **celular** | `baixar_scin.py` |
| ISIC 2018 Task 3 | HAM10000 com o teste oficial (1.512) | — | — | dermatoscópio | `baixar_dados.py isic2018` (não entra no manifesto: o treino já está dentro do ISIC 2019 e o teste é lido direto do zip por `treinar_professor.py`) |

Licenças: as bases da ISIC e a MILK10k são CC BY-NC 4.0 (PAD-UFES-20 é CC BY 4.0), MED-NODE é
para uso acadêmico e a SCIN tem a própria *SCIN Data Use License* (baixada junto). Só uso
não comercial.

Para reproduzir:

```
py -3.12 scripts/baixar_dados.py                 # ISIC 2018/2019/2020/2024 e MILK10k (retomável)
py -3.12 scripts/baixar_metadados_clinicas.py    # lista as fotos clínicas na API da ISIC
py -3.12 scripts/baixar_imagens_clinicas.py      # baixa as fotos (PAD-UFES-20, HIBA, MSKCC, anônima)
py -3.12 scripts/baixar_scin.py                  # SCIN (11,7 GB)
uv run scripts/preparar_dados.py [base ...]      # redimensiona para 256 px e gera o manifesto
uv run scripts/avaliar_qualidade.py              # nitidez, brilho, quase-duplicatas, mosaicos por fonte
```

`preparar_dados.py` gera `data/processed/imagens_256/<base>/<id>.jpg` (menor lado 256 px) e o
`manifesto.csv`, uma linha por imagem: `classe` harmonizada entre as bases (`MEL NV BCC SCC AK BKL
DF VASC INF OUTRO_BENIGNO OUTRO_MALIGNO BENIGNO_SEM_DIAGNOSTICO INDETERMINADO UNK`), `melanoma`,
`maligno`, `grupo` (paciente ou lesão: imagens do mesmo grupo nunca se separam entre treino,
validação e teste), `split` (oficial quando a base tem teste oficial, senão sorteado com semente
fixa), `duplicada_de` (mesma imagem em outra base; descartar ao juntar), idade, sexo, região do
corpo, tom de pele e fonte. Cada base fica em cache em `data/processed/_cache/`, então dá para
processar uma de cada vez; o ISIC 2024 precisa de memória livre (16 GB foram apertados). O
manifesto do repositório é descompactado com `gzip -dk data/processed/manifesto.csv.gz` (ou
`pd.read_csv("data/processed/manifesto.csv.gz")` direto).

Os scripts `testar_modelo.py` (avalia um `.keras` em qualquer base do manifesto, agrupando por
origem) e `treinar_simulando_celular.py` (treino do modelo) usam esse manifesto e as imagens em
`imagens_256/`.

### Treino no Kaggle (arquivado)

Nesta fase também foi preparado um caminho para treinar redes maiores na GPU do Kaggle, que nunca
chegou a rodar. Ele ficou em `arquivo/kaggle/`, com a explicação no README da pasta.

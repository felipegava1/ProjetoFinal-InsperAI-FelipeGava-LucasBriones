# Projeto Final – InsperAI (Felipe Gava, Lucas Briones)

## Classificador de dermatoscopia (PyTorch) — escopo atual

> O projeto passou a focar só em **dermatoscopia** (ver o "Plano de iterações"). O código novo está
> em `dermato/`; o restante deste README descreve a versão anterior (TensorFlow, foto de celular).
> Resultado = apoio à priorização (quem precisa de biópsia/encaminhamento primeiro), nunca diagnóstico.

**Resultados na validação** (5.821 imagens, TTA 8; os testes T1/T2 ainda não foram abertos):

| Modelo | Resolução | Acurácia balanceada | Especificidade a 95% de sensibilidade | AUC melanoma |
|---|---|---|---|---|
| E0 — receita atual (EfficientNetV2-B0) | 224 | 0,696 | 0,689 | 0,912 |
| E0_busca — B0 com parâmetros otimizados | 224 | 0,7345 | 0,703 | 0,917 |
| **E1 — ConvNeXt-Tiny** | **384** | **0,769** | **0,773** | **0,933** |

Raciocínio, mudanças e números de cada etapa: `resultados/DIARIO.md`; uma linha por treino em
`resultados/experimentos.csv`; todas as tentativas das buscas em `resultados/optuna.db`.

### Pesos dos modelos e Git LFS

Os arquivos de pesos (`resultados/modelos/*.pt`, de 23 a 107 MB) são grandes demais para o git
comum (o GitHub recusa arquivos acima de 100 MB). Por isso eles ficam no **Git LFS** (Large File
Storage): o repositório guarda só um "ponteiro" de poucos bytes e o arquivo de verdade fica no
armazenamento LFS do GitHub, baixado junto com o clone.

| Arquivo | Tamanho | Modelo |
|---|---|---|
| `E0_final_s42.pt` | 23 MB | E0, receita atual (EfficientNetV2-B0, 224 px) |
| `E0_busca_final_s42.pt` | 23 MB | E0 com parâmetros otimizados |
| `E1_final_s42.pt` | 107 MB | E1, ConvNeXt-Tiny 384 px (melhor modelo) |

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

**Conferir se os pesos estão íntegros:** `git lfs ls-files` lista os três arquivos; os códigos devem
bater com `resultados/modelos_sha256.txt` (no Windows: `certutil -hashfile resultados\modelos\E1_final_s42.pt SHA256`).

**Cota:** a conta gratuita do GitHub tem 1 GB de armazenamento e 1 GB de download por mês para
arquivos LFS (do dono do repositório). Os três pesos somam ~153 MB, ou seja, cerca de 6 clones
completos por mês. Para clonar sem baixar os pesos: `GIT_LFS_SKIP_SMUDGE=1 git clone ...`
(no PowerShell: `$env:GIT_LFS_SKIP_SMUDGE=1` antes do clone).

**Adicionar um modelo novo** (ex.: depois de treinar a E2): copie o `.pt` para `resultados/modelos/`
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

Projeto acadêmico de apoio à triagem de lesões de pele; o resultado é sempre uma sugestão ou
orientação, nunca um diagnóstico. Todo o trabalho (código e texto) está no `notebook.ipynb`, e o
relatório completo em `relatorios/relatorio_projeto.pdf`.

Um **único modelo de pintas** (EfficientNetV2-B0, 224 px, com idade, sexo e região do corpo) atende
foto de celular e dermatoscopia. A ideia central: ensinar a rede a reconhecer lesões em foto de celular
**sem usar nenhuma foto de celular no treino**, com fotos clínicas comuns e uma augmentation que simula
celular (`pipeline.simular_celular`).

| Teste | Acurácia | Acurácia balanceada | AUC melanoma | AUC maligno × benigno |
|---|---|---|---|---|
| Dermatoscopia (ISIC 2018, 1.511 imagens) | 82,8% | 75,0% | 0,931 | 0,945 |
| Foto de celular (PAD-UFES-20 + HIBA, 2.643 fotos) | 38,9% | 47,8% | 0,871 | 0,868 |

A triagem de erupções, alergias e queimaduras está **pausada** e separada em `triagem_alergias/`.

## Preparando a base de dados

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

### Estrutura gerada

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

### Usando no notebook

```python
from prepare_data import load_labels, CLASSES
df = load_labels()  # DataFrame com caminhos absolutos
```

O `notebook.ipynb` já carrega a base e monta os datasets do TensorFlow na primeira seção.
O carregamento em `tf.data` fica em `pipeline.py`, o mesmo código usado no Kaggle.

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

## Bases de dados ampliadas (fotos de celular)

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

## Treinando no Kaggle (arquiteturas maiores)

Caminho alternativo, ainda sem a simulação de celular: serve para testar redes maiores na GPU grátis do
Kaggle (~30 h/semana), a partir do ISIC 2019. O modelo do projeto não saiu daqui. As imagens de
treino/validação vêm do dataset público
[andrewmvd/isic-2019](https://www.kaggle.com/datasets/andrewmvd/isic-2019); o resto sobe como
dois datasets privados.

1. **Conta**: crie uma conta no Kaggle e verifique o telefone (Settings → Phone verification),
   necessário para usar GPU e Internet nos notebooks.
2. **Pacotes**: com a base local pronta, rode
   ```
   python kaggle_package.py SEU_USUARIO_KAGGLE
   ```
   Isso cria em `~/datasets/kaggle/`:
   - `lesoes-pele-meta/`: `labels.csv`, `md5.csv` e o código (`prepare_data.py`, `pipeline.py`)
   - `lesoes-pele-extra/`: imagens de `test/` e `unlabeled/` (~3,8 GB)
3. **Upload** (os datasets são criados como privados):
   ```
   pip install kaggle
   kaggle datasets create -p ~/datasets/kaggle/lesoes-pele-meta
   kaggle datasets create -p ~/datasets/kaggle/lesoes-pele-extra --dir-mode zip
   ```
   O `kaggle` precisa do token da API: Kaggle → Settings → API → *Create New Token*, e salve o
   `kaggle.json` em `C:\Users\<usuario>\.kaggle\`. Como alternativa, dá para subir as pastas pelo
   site (Datasets → New Dataset).
4. **Notebook**: no Kaggle, Create → New Notebook → File → *Import Notebook* →
   `kaggle_professor.ipynb`. Em Settings, escolha *Accelerator: GPU T4 x2* e ligue *Internet*; em
   *Add Input*, anexe `andrewmvd/isic-2019`, `lesoes-pele-meta` e `lesoes-pele-extra`.
   A seção Dados confere se todas as imagens foram encontradas e se são idênticas às locais (md5).
5. **Mudou o código?** Se `prepare_data.py` ou `pipeline.py` mudarem, rode o passo 2 de novo e
   publique uma nova versão: `kaggle datasets version -p ~/datasets/kaggle/lesoes-pele-meta -m "descrição"`.

### Treinando o professor

O `kaggle_professor.ipynb` treina um ou mais professores (`MODELS`, na seção Modelo professor),
avalia no teste e exporta modelo e previsões. Use **Save Version → Save & Run All**, que roda em
segundo plano (até 12 h por execução) e guarda tudo na aba Output.

| Professor | Arquitetura | Tempo estimado (T4 x2) |
|---|---|---|
| `effv2s` | EfficientNetV2-S, 384 px | ~2 h |
| `effb4` | EfficientNet-B4, 380 px | ~3 h |
| `effv2m` | EfficientNetV2-M, 384 px | ~4 h |
| `convnext_s` | ConvNeXt-Small, 384 px | ~5 h |

Roteiro sugerido:
1. Rode com `MODELS = ["effv2s"]` (o padrão). Isso valida tudo de ponta a ponta e já dá o primeiro professor.
2. Para o **ensemble**, rode novas versões com os outros professores (ex.: `MODELS = ["convnext_s"]`)
   e, em *Add Input*, anexe o **output das versões anteriores** (Your Work → Notebooks). A seção
   Teste encontra todos os `probs.npz`, avalia cada professor e a média deles, e escolhe o melhor
   pela validação.
3. Se faltar memória na GPU (erro OOM), reduza o `batch` do professor em `BACKBONES`; se der erro
   de `MirroredStrategy`, use `MULTI_GPU = False`.

O output de cada versão traz `<professor>/melhor.keras` (modelo), `<professor>/probs.npz`
(probabilidades com TTA de todos os splits), as curvas de treino, `resultados.csv` e
`ensemble_probs.npz` (probabilidades do melhor, para a destilação do aluno).

Para testar o notebook no PC antes de subir (poucas imagens, resolução baixa, uma época), rode-o
com as variáveis de ambiente `SMOKE_TEST=1`, `KAGGLE_INPUT` (pasta com os três datasets),
`KAGGLE_WORKING` e `CACHE_DIR`.

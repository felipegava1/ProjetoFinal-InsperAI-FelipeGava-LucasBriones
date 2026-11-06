# Projeto Final – InsperAI (Felipe Gava, Lucas Briones)

Classificação de lesões de pele com uma CNN professor-aluno, usando o ISIC 2019
(HAM10000 + BCN_20000 + MSK) para treino e o ISIC 2018 Task 3 para teste.

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

## Treinando no Kaggle

O professor é treinado no Kaggle, que tem GPU grátis (~30 h/semana). As imagens de
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

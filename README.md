# Projeto Final – InsperAI (Felipe Gava, Lucas Briones)

Classificação de lesões de pele com o dataset HAM10000 / ISIC 2018 Task 3.

## Preparando a base de dados

As imagens (~3 GB) ficam só no seu PC, fora do repositório e fora do OneDrive.

1. Baixe o `dataverse_files.zip` do Harvard Dataverse:
   https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
   (botão *Access Dataset* → *Download ZIP*).
2. Instale as dependências: `pip install pandas tensorflow`
3. Rode, na pasta do repositório:
   ```
   python organize_dataset.py CAMINHO/PARA/dataverse_files.zip
   ```
   Também aceita a pasta onde o zip já foi extraído. Por padrão a base vai para
   `~/datasets/ham10000` (no Windows, `C:\Users\<usuario>\datasets\ham10000`). Para outro
   lugar, passe o destino como segundo argumento e defina a variável de ambiente
   `HAM10000_DIR` com esse caminho, para que o `load_labels()` o encontre.
4. Depois de pronto, o `dataverse_files.zip` pode ser apagado.

### Estrutura gerada

```
ham10000/
├── train/<dx>/ISIC_xxxxxxx.jpg   8502 imagens (HAM10000 menos a validação)
├── val/<dx>/ISIC_xxxxxxx.jpg     1513 imagens (~15% das lesões, estratificado por classe)
├── test/<dx>/ISIC_xxxxxxx.jpg    1511 imagens (ISIC 2018 Task 3 Test)
├── masks/                        máscaras de segmentação (só train/val)
└── labels.csv                    metadados + split + caminhos relativos
```

`<dx>` é a classe: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`. A coluna `malignant` do
`labels.csv` vale 1 para `mel`, `bcc` e `akiec`.

A validação é sorteada por `lesion_id` (semente fixa), então fotos da mesma lesão nunca ficam
em train e val ao mesmo tempo. A imagem `ISIC_0035068` está no gabarito de teste mas não vem
no zip, por isso é descartada.

### Usando no notebook

```python
from organize_dataset import load_labels, DEFAULT_DIR
import keras

df = load_labels()  # DataFrame com caminhos absolutos

train_ds = keras.utils.image_dataset_from_directory(
    DEFAULT_DIR / "train", image_size=(224, 224), batch_size=32)
```

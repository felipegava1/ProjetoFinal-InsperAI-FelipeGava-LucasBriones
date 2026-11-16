# Treino no Kaggle (arquivado, nunca executado)

> **Status: arquivado.** Este caminho foi preparado em 06/11/2026, na fase do modelo v1 (TensorFlow), e
> **nunca chegou a rodar**: o notebook não tem saídas e nenhum modelo do projeto saiu daqui. O plano
> era treinar "professores" grandes na GPU grátis do Kaggle e destilar o conhecimento num "aluno"
> menor (pseudo-rótulos nas 6.726 imagens sem rótulo do ISIC 2019). Ele perdeu o sentido quando o
> projeto passou para PyTorch com GPU local: a E1 (ConvNeXt-Tiny, 384 px, ver
> `notebook_dermatoscopia.ipynb`) cumpriu o papel de "rede maior em resolução maior".
>
> O professor que existiu de fato (`professor_effv2b0`, iteração 1 do `notebook_v1_celular.ipynb`)
> foi treinado na CPU local, não no Kaggle.

Ficam aqui só como registro:

- `kaggle_professor.ipynb`: treina um ou mais professores, avalia no teste do ISIC 2018 (inclusive
  contra dermatologistas) e exporta modelo e previsões.
- `kaggle_package.py`: monta os datasets privados a subir no Kaggle a partir da base local do
  `prepare_data.py` (ver o README da raiz). Usa `prepare_data.py` e `pipeline.py` da raiz do repositório.

## Como seria usado

Caminho sem a simulação de celular, para testar redes maiores na GPU grátis do Kaggle (~30 h/semana),
a partir do ISIC 2019. As imagens de treino/validação vêm do dataset público
[andrewmvd/isic-2019](https://www.kaggle.com/datasets/andrewmvd/isic-2019); o resto sobe como dois
datasets privados.

1. **Conta**: crie uma conta no Kaggle e verifique o telefone (Settings → Phone verification),
   necessário para usar GPU e Internet nos notebooks.
2. **Pacotes**: com a base local pronta, rode na raiz do repositório
   ```
   python arquivo/kaggle/kaggle_package.py SEU_USUARIO_KAGGLE
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

Roteiro previsto:
1. Rodar com `MODELS = ["effv2s"]` (o padrão), para validar tudo de ponta a ponta e já ter o primeiro professor.
2. Para o **ensemble**, rodar novas versões com os outros professores (ex.: `MODELS = ["convnext_s"]`)
   e, em *Add Input*, anexar o **output das versões anteriores** (Your Work → Notebooks). A seção
   Teste encontra todos os `probs.npz`, avalia cada professor e a média deles, e escolhe o melhor
   pela validação.
3. Se faltar memória na GPU (erro OOM), reduzir o `batch` do professor em `BACKBONES`; se der erro
   de `MirroredStrategy`, usar `MULTI_GPU = False`.

O output de cada versão traria `<professor>/melhor.keras` (modelo), `<professor>/probs.npz`
(probabilidades com TTA de todos os splits), as curvas de treino, `resultados.csv` e
`ensemble_probs.npz` (probabilidades do melhor, para a destilação do aluno).

Para testar o notebook no PC antes de subir (poucas imagens, resolução baixa, uma época), rode-o
com as variáveis de ambiente `SMOKE_TEST=1`, `KAGGLE_INPUT` (pasta com os três datasets),
`KAGGLE_WORKING` e `CACHE_DIR`.

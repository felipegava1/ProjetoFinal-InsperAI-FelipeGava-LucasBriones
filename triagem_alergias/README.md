# Triagem de erupções, alergias e queimaduras (pausada)

Parte do projeto pausada para focar no modelo de pintas. Tudo dela está nesta pasta, e o notebook é
`triagem_alergias.ipynb`, que deve ser aberto com esta pasta como diretório de trabalho. Ele usa `pipeline.py`,
`prepare_data.py` e `modelos/pintas/` da raiz do projeto.

## Triagem em duas etapas

```
imagem -> detector de tipo de foto -+- dermatoscopia -> modelo de pintas
                                    +- foto de celular -> roteador -+- pinta -> modelo de pintas + orientação de avaliação médica
                                                                    +- erupção/alergia -> especialista (17 classes)
                                                                    +- baixa confiança -> "não foi possível identificar"
```

Na foto de celular, o modelo sugere as classes mais prováveis, mas a urgência nunca cai para "cuidados em
casa": foto de celular não permite descartar um câncer.

Para rodar, além da base do README principal:

1. Coloque os parquets do SCIN na pasta `data_Alergias`, ao lado do repositório (ou defina `SCIN_RAW`).
2. Instale as dependências extras: `pip install pyarrow imagehash pyyaml scipy` (a primeira célula do notebook já faz isso).
3. Execute o notebook. O PAD-UFES-20 (cerca de 3,4 GB) é baixado automaticamente. As imagens tratadas ficam em
   `~/datasets`, e cada rede só é treinada se não estiver em `modelos/triagem/`.

| Etapa no notebook | Saída |
|---|---|
| 1. Exploração do SCIN | conclusões e gráficos no próprio notebook |
| 2. Limpeza | `~/datasets/scin/` e `relatorios/limpeza_scin.md` |
| 3. Treino, validação e teste | `splits/scin_v1.csv`, `splits/pad_v1.csv` |
| 4. Arquitetura e treino | `modelos/triagem/*.keras`, `limiares.json` |
| 5. Orientação | `config/orientacoes.yaml` (rascunho a revisar por um profissional de saúde) e a função `triagem(imagem, respostas_alerta)` |
| 6. Avaliação | `relatorios/metricas_triagem.json`, `relatorios/figuras/` |
| 7. Especialista v2 (modelos de base congelados + metadados) | `experimentos/*/metricas.json`, `relatorios/especialista_v2.md`, `relatorios/figuras/especialista_v2_confusao.png` |

Limitações principais: o roteador aprendeu em boa parte a origem das fotos (reconhece só 5,7% das pintas do SCIN),
o especialista é fraco para indicar uma única condição, e não há dados suficientes de queimaduras e feridas. O erro
grave (orientar cuidados em casa quando precisaria de atendimento) é de 13,7%: 8,5% em dermatoscopia e 31,8% nas fotos do SCIN.

## Especialista v2 (etapa 7, experimental)

Em vez de ajustar uma CNN pequena, a v2 usa embeddings congelados de modelos de base (DINOv2-base, SigLIP-base) com uma
cabeça leve treinada com rótulo suave, mais a média dos embeddings das fotos do mesmo caso e 71 metadados autodeclarados do
paciente. No teste (802 fotos, 17 classes) a melhor cabeça sobe de 28,1% para 39,0% em top-1 e de 17,6% para 31,2% em
acurácia balanceada; a média com o especialista da etapa 4 chega a 40,3% top-1, 73,9% top-3 e AUROC 0,839. O teto imposto
pela discordância dos dermatologistas da SCIN é de 86% em top-1 (só 31% das fotos têm um único diagnóstico).

Os scripts ficam em `scripts/` e precisam das imagens da SCIN preparadas pela raiz do projeto
(`scripts/baixar_scin.py` e `scripts/preparar_dados.py`, pasta `data/processed/imagens_256/scin/`):

```bash
uv run triagem_alergias/scripts/extrair_embeddings.py dinov2_base      # ~25 min na CPU; também siglip_base, monet
uv run triagem_alergias/scripts/treinar_cabeca.py --embeddings dinov2_base --meta --caso --arq mlp
uv run triagem_alergias/scripts/avaliar_felipe.py                      # especialista da etapa 4 nas mesmas fotos
py -3.12 triagem_alergias/scripts/combinar.py felipe dinov2_base+caso+meta_suave_mlp --nome combinacao_felipe+dinov2mlp
py -3.12 triagem_alergias/scripts/resumir.py test
```

Cada experimento grava `probs_val.npy`, `probs_test.npy` e `metricas.json` em `experimentos/<nome>/`; a melhor cabeça
(`experimentos/dinov2_base+caso+meta_suave_mlp/cabeca.pt`) ainda não está ligada à função `triagem`. Os embeddings
(`experimentos/embeddings/*.npy`, 48 MB) ficam fora do repositório.

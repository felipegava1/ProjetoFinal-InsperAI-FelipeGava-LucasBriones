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

Limitações principais: o roteador aprendeu em boa parte a origem das fotos (reconhece só 5,7% das pintas do SCIN),
o especialista é fraco para indicar uma única condição, e não há dados suficientes de queimaduras e feridas. O erro
grave (orientar cuidados em casa quando precisaria de atendimento) é de 13,7%: 8,5% em dermatoscopia e 31,8% nas fotos do SCIN.

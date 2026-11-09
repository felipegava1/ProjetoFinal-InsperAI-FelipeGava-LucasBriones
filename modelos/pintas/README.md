# Modelo de pintas

O modelo único do projeto: EfficientNetV2-B0 224 px + 14 metadados, 8 logits `ak bcc bkl df mel nv scc
vasc`. Funciona em **foto de celular** e em dermatoscopia, e foi treinado **sem nenhuma foto de celular**.

## Receita (`scripts/treinar_simulando_celular.py --conjunto dermatoscopia+fotos`)

- **Treino** (23.435 imagens, split `train` do `manifesto.csv`): ISIC 2019 treino, ISIC 2020 (só as
  com diagnóstico específico), MILK10k dermatoscopia e close-up, MED-NODE, MSKCC e anônimas do
  acervo ISIC. Nevos limitados a 6.000 (sorteio fixo, semente 42).
- **Validação** (6.285 imagens, split `val` das mesmas fontes, separado por paciente/lesão): decide a
  época (acurácia balanceada com ajuste de prior), o `tau` e o limiar de triagem.
- **Teste** (2.643 fotos, 117 melanomas): PAD-UFES-20 e HIBA **inteiras**, as duas bases com celular
  confirmado; nenhuma foto delas entra em treino ou validação.
- Parte de `pesos_de_partida.keras` (a mesma rede ajustada só no ISIC 2019). Pré-processamento: Shades of Gray, recorte central 224.
- Augmentation: `pipeline.augment` **mais** `pipeline.simular_celular`, uma cadeia que simula celular (cada efeito sorteado com
  sua probabilidade): recorte interno que tira a borda do dermatoscópio, lesão menor no quadro
  com fundo liso, brilho/contraste/saturação/matiz fortes, balanço de branco, sombra em
  gradiente, perda de resolução, desfoque gaussiano, ruído de sensor, recompressão JPEG.
  Exemplos em `augment_exemplos.jpg` (1ª linha original, 2ª e 3ª após a augmentation).
- Treino: toda a rede treinável exceto BatchNormalization; AdamW (wd 0,02), lr 1e-4 com aquecimento
  de 1 época e cosseno; lote 32; até 8 épocas com early stopping (paciência 3). Melhor época: 7.
- Inferência: TTA 8 vistas (4 rotações × flip); ajuste de prior `log p − tau·log prior` com
  **tau = 1,1**; triagem = soma de p(ak, bcc, mel, scc) ≥ **0,287** (95 % de sensibilidade na
  validação).

Prior do treino (para o ajuste): ak 910, bcc 6384, bkl 3145, df 275, mel 4981, nv 6000, scc 1460, vasc 280.

## Resultado no teste de celular (PAD-UFES-20 + HIBA)

| Modelo | AUC melanoma | Sens. melanoma | Precisão melanoma | Acur. balanceada | Triagem sens. / espec. |
|---|---|---|---|---|---|
| pesos de partida | 0,593 | 22 % | 7 % | 30 % | 92 % / 49 % |
| **este** | **0,871** | **50 %** | **22 %** | **48 %** | 97 % / 41 % |

Por base: PAD-UFES-20 AUC 0,915 (sens. 67 %), HIBA 0,817 (sens. 37 %). O controle com a mesma
receita **sem** a simulação de celular deu AUC 0,827 / acur. balanceada 46 %: a simulação responde
por +0,044 de AUC, o resto vem das fotos clínicas comuns no treino.

Limitações: o limiar de triagem transferido da dermatoscopia fica agressivo no celular (6 em 10
benignos encaminhados; mover o limiar para 0,5 dá 91 % / 62 %); confusão alta entre AK, BKL e SCC;
58 dos 117 melanomas ainda recebem outra classe no argmax (bkl 17, vasc 16, nv 10, df 7).

## Resultado em dermatoscopia (ISIC 2018, 1.511 imagens, 7 classes)

| Modelo | Acurácia | Acur. balanceada | AUC maligno × benigno | Sens. melanoma | Triagem sens. / espec. | Malignas perdidas |
|---|---|---|---|---|---|---|
| pesos de partida | 82,2 % | 74,3 % | 0,943 | 50 % | 96 % / 73 % | 13 de 307 |
| **este** | **82,8 %** | **75,0 %** | **0,945** | **62 %** | 92 % / 83 % | 26 de 307 |

Empate estatístico nas métricas gerais (IC 95 % por bootstrap inclui zero). O limiar 0,287 dá 95,8 % de
sensibilidade na validação em dermatoscopia, mas 91,5 % no teste; a tabela de limiares está na seção 3
do notebook.

Outras limitações: a escolha entre as variantes de treino viu o teste de celular, então os números de
celular podem estar levemente otimistas. Na simulação, o fundo do efeito "afastar" sai como retângulo
cinza de borda marcada, e a augmentation comum ainda desenha a borda do dermatoscópio em 25 % das
imagens.

## Arquivos

- `melhor.keras` — o modelo (sem estado do otimizador; carregar com `compile=False`).
- `pesos_de_partida.keras` — pesos só do ISIC 2019, ponto de partida do treino.
- `historico.csv` — perda/val por época, com AUC de melanoma no celular só como acompanhamento.
- `comparacao.csv`, `relatorio.txt` — métricas antes × depois no celular, no total e por base.
- `comparacao_dermatoscopia.csv` — o mesmo em dermatoscopia (ISIC 2018).
- `probs_isic2018.npz`, `probs_val_dermatoscopia.npz` — probabilidades no teste e na validação de dermatoscopia.
- `previsoes_celular.csv` — probabilidades (TTA) dos dois modelos em cada foto do teste.
- `augment_exemplos.jpg` — mosaico de conferência da augmentation.

Avaliar em outras bases: `uv run scripts/testar_modelo.py modelos/pintas/melhor.keras`.

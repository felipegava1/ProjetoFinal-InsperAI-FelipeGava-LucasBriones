# Relatório de limpeza do SCIN

Gerado pela Etapa 2 da Parte 2 do `notebook.ipynb`. Os números vêm de `~/datasets/scin/limpeza.json`.

**Atenção:** a pasta `data_Alergias` tem 25 dos 26 parquets; falta `train-00000-of-00026.parquet`.

## Entrada e saída por etapa

| Etapa | Imagens removidas | Imagens restantes | Casos restantes |
|---|---:|---:|---:|
| Entrada (extraídas dos parquets) | | 9.998 | 4.839 |
| Corrompidas ou ilegíveis | 0 | 9.998 | 4.839 |
| Muito pequenas (lado menor < 224 px) | 26 | 9.972 | 4.838 |
| Duplicatas exatas (MD5) | 25 | 9.947 | 4.820 |
| Quase-duplicatas (pHash + correlação) | 15 | 9.932 | 4.818 |
| Duplicatas com as bases de pintas | 0 | 9.932 | 4.818 |
| Não avaliáveis pelos dermatologistas | 3.782 | 6.150 | 2.942 |
| Sem rótulo | 175 | 5.975 | 2.847 |
| Confiança muito baixa | 541 | **5.434** | **2.581** |

A documentação do SCIN informa 15 imagens duplicadas, que aparecem 42 vezes. Aqui foram 15 imagens em 40 ocorrências, coerente com a falta de um arquivo.

## Decisões e limiares

**Quase-duplicatas (`NEAR_DUP_MAX = 12`, `NEAR_DUP_CORR = 0,98`).** Só o pHash não bastou. Na revisão manual de amostras de pares, fotos de pele de pessoas diferentes já apareciam com distância 4 ou 6, principalmente na comparação com a imagem espelhada, porque fotos de pele têm pouco detalhe. Por isso o pHash só gera candidatas (distância ≤ 12, direta ou espelhada), e a confirmação vem da correlação dos pixels em cinza 64×64:

| Correlação | O que os pares revisados mostraram |
|---|---|
| ≥ 0,98 | sempre a mesma foto (3 entre casos diferentes, 16 no mesmo caso) |
| 0,96 a 0,98 | fotos em sequência do mesmo caso; um par de pessoas diferentes (r = 0,970) |
| 0,93 a 0,96 | pessoas diferentes |

Limitação: uma foto recortada de outra pode passar despercebida, porque a correlação cai com o recorte.

**Duplicatas entre bases.** Cada imagem do SCIN foi comparada com as 33.568 imagens das bases de pintas (ISIC 2019, ISIC 2018 e HAM10000), por MD5 e pelo mesmo critério. Nenhuma coincidiu. O PAD-UFES-20 não está neste computador e não foi verificado.

**Não avaliável.** Um caso sai quando há mais votos de "qualidade insuficiente" do que votos de avaliável entre seus dermatologistas.

**Confiança (`MIN_CONFIDENCE = 1,5`).** A confiança vai de 1 a 5. O limiar remove os casos em que os dermatologistas estavam, em média, quase no mínimo de confiança. Um limiar de 2,0 removeria bem mais casos, em uma base já pequena.

**Sinônimos.** Os pesos são somados antes de escolher a condição principal:
- `Allergic Contact Dermatitis`, `Irritant Contact Dermatitis` e `CD - Contact dermatitis` → `Contact Dermatitis`. Foto nenhuma distingue alérgica de irritativa com segurança.
- `Acute dermatitis, NOS`, `Chronic dermatitis, NOS` e `Acute and chronic dermatitis` → `Dermatitis, NOS`.

**Condição principal.** É a de maior peso. Nos empates, fica a condição mais frequente na base e, persistindo o empate, a primeira em ordem alfabética. A coluna `rotulos_suaves` guarda todos os pesos, para comparar com o treino com rótulos suaves.

**Classes raras (`MIN_CASES = 30`).** As 145 condições com menos de 30 casos, somando 642 casos, foram agrupadas em `outras_condicoes`, em vez de descartadas. Descartar perderia 25% dos casos e faria o modelo escolher sempre uma das condições comuns, mesmo diante de algo que não conhece. Com a classe "outras", a saída pode ser honesta: a lesão não parece nenhuma das condições conhecidas.

## Resultado

| Grupo | Casos |
|---|---:|
| `erupcao_alergia` | 2.530 |
| `ferida` | 30 |
| `pinta` | 17 |
| `queimadura` | 4 |

O grupo vem da condição principal, por termos definidos em `GROUP_TERMS`, no notebook. O grupo `erupcao_alergia` inclui, por enquanto, tudo o que não é ferida, queimadura ou pinta, como infecções (tinea, herpes, impetigo).

| Condição | Casos |
|---|---:|
| Eczema | 495 |
| Contact Dermatitis | 442 |
| Urticaria | 181 |
| Insect Bite | 164 |
| Folliculitis | 114 |
| Psoriasis | 77 |
| Dermatitis, NOS | 65 |
| Tinea | 63 |
| Herpes Simplex | 58 |
| Acne | 50 |
| Herpes Zoster | 45 |
| Pigmented purpuric eruption | 44 |
| Drug Rash | 42 |
| Pityriasis rosea | 35 |
| Tinea Versicolor | 34 |
| O/E - ecchymoses present | 30 |
| `outras_condicoes` | 642 |

Os splits estão em `splits/scin_v1.csv`, gerados e verificados na Etapa 3 do `notebook.ipynb`.

# Diário de bordo — treino de E0, E1 e E2

Registro de cada decisão, mudança e raciocínio, na ordem em que aconteceram. Os números de cada
tentativa ficam em `experimentos.csv` (uma linha por rodada) e em `optuna.db` (todas as tentativas
das buscas, com parâmetros). Tudo que é escolhido (época, parâmetros, tau, limiar) vem só da
**validação**; os testes T1/T2 não são abertos nesta fase.

Máquina: i5-14600K (20 threads), 32 GB RAM, RTX 5070 12 GB, Windows 11. PyTorch 2.11 + CUDA 12.8, timm 1.0.

---

## 12/11 — Preparação (antes de qualquer treino)

**Dados.** Divisão fixa em `splits/dermato.csv` (semente 42), conforme o plano:

| Parte | Imagens | Origem |
|---|---|---|
| treino | 32.980 | ISIC 2019 treino, metade A do teste ISIC 2019, ISIC 2020 (diagnóstico específico), MILK10k dermatoscopia |
| validação | 5.821 | 15% do conjunto acima, sorteado por grupo (paciente/lesão), estratificado por classe |
| T1 | 1.511 | ISIC 2018 Task 3 teste (7 classes HAM: akiec = ak + scc) |
| T2 | 2.340 | metade B do teste ISIC 2019, sem as imagens do ISIC 2018 |

Decisões e porquês:
- **ISIC 2020: só as 5.993 imagens usadas**, baixadas uma a uma do ISIC Archive (1,6 GB), em vez do
  zip oficial de 24,7 GB (que estava baixando a 3–18 MB/s). As outras 27 mil não entram no projeto
  (diagnóstico "unknown"). Menor lado ≥ 480 px, então servem para 384 px.
- **Teste do ISIC 2019 sem `lesion_id`**: ali o grupo é a própria imagem. Limitação: duas fotos da
  mesma lesão do BCN podem cair uma no treino e outra no T2. Mesmo problema no ISIC 2019 treino para
  BCN/MSK (só o HAM informa lesão) — herdado do projeto original.
- **ISIC 2020: grupo = paciente**; MILK10k: grupo = lesão; duplicatas declaradas do ISIC 2020 removidas.
- **Cache** em 448 px (menor lado), cores originais. Shades of Gray, recorte e augmentation são
  aplicados na hora porque são parâmetros da busca ("com e sem" SoG).

**Código.** Pipeline novo em PyTorch em `dermato/` (preparar, dados, modelos, metricas, treinar, busca,
fila). Validado com testes de fumaça (poucas imagens, 2 épocas) antes de qualquer treino real.

**Velocidade medida na GPU** (treino bf16, dados sintéticos): B0 224 px 1.088 img/s; ConvNeXt-Tiny
224/288/384 px: 702/420/242 img/s; MaxViT-Tiny 224 px 203 img/s, 384 px 94 img/s. Memória sempre
abaixo de 8 GB.

- **Decisão: E2 (MaxViT) fixa em 224 px.** A 384 px seriam ~4 min por época, ~1 h por tentativa: as
  30 tentativas não caberiam. O próprio plano nomeia `maxvit_tiny_tf_224`.

**Carregamento como gargalo.** A primeira época do E0 levou ~1,1 min (~340 img/s), um terço do que a
GPU aguenta. Mudanças:
- Shades of Gray passa a estimar a iluminação com 1/16 dos pixels (1 a cada 4 em cada direção).
  Conferido: diferença média de 0,03 nível de cor (em 255) em relação à versão completa.
- 16 processos de carregamento em vez de 10.
- Como isso muda (minimamente) o pré-processamento, o E0 que estava rodando foi descartado e
  recomeçado, para todas as etapas usarem exatamente o mesmo pré-processamento (regra 5 do protocolo).

**E0 em duas versões.** O plano define o E0 como a receita atual fixa (a régua). O pedido agora é
maximizar os três modelos, então:
- `E0_final_s42` — **receita atual**, sem mudar nada (régua de comparação);
- `E0_busca` — a mesma rede (EfficientNetV2-B0) com busca Optuna de 30 tentativas, incluindo
  resolução 224/288/384 px. É barata (~20 s por época) e separa "ganho da rede nova" de "ganho de
  ajustar parâmetros" na comparação com E1 e E2.

Receita do E0 (portada de `scripts/treinar_simulando_celular.py`, sem a simulação de celular):
EfficientNetV2-B0 (`tf_efficientnetv2_b0.in1k`, ImageNet), 224 px, 14 metadados, Shades of Gray,
augmentation do `pipeline.augment` (recorte 50–100%, flips, rotações de 90°, cor leve, borda do
dermatoscópio em 25%), AdamW lr 1e-4, weight decay 0,02, 1 época de aquecimento + cosseno, lote 32,
8 épocas, paciência 3, BatchNorm congelada, nevos limitados a 6.000, entropia cruzada + ajuste de
prior (tau escolhido na validação). Diferença inevitável: o Keras original partia de pesos já
ajustados no ISIC 2019; aqui parte do ImageNet, como pede o plano.

---

## 12/11 21:41–21:51 — E0 (receita atual): a régua

Treino de 9,5 min (8 épocas; ~0,8 min por época depois da primeira, que inclui ~3 min de partida).

| Época | perda | val bal_acc | AUC mel | spec95 |
|---|---|---|---|---|
| 1 | 1,263 | 0,553 | 0,843 | 0,565 |
| 2 | 0,905 | 0,647 | 0,859 | 0,568 |
| 4 | 0,675 | 0,660 | 0,899 | 0,662 |
| 6 | 0,505 | 0,678 | 0,897 | 0,658 |
| 8 | 0,418 | 0,687 | 0,907 | 0,665 |

**Resultado final na validação (melhor época = 8, TTA 8 vistas, tau 1,45):** acurácia balanceada
**0,696**, especificidade a 95% de sensibilidade **0,689**, AUC melanoma 0,912, AUC maligno × benigno
0,933, ECE 0,056.

Leitura:
- **A curva ainda subia na época 8**: com pesos do ImageNet, 8 épocas são poucas (o Keras original
  partia de pesos já ajustados no ISIC 2019, por isso 8 bastavam lá). É a primeira hipótese a testar
  na busca do E0: mais épocas (a faixa vai até 15).
- tau alto (1,45): o modelo treinado com a distribuição natural (muitos nevos) precisa de um desconto
  forte do prior para a acurácia balanceada. Perdas ponderadas/amostragem balanceada (na busca) devem
  reduzir essa dependência.
- A especificidade a 95% (0,689) é menor que os 83,2% citados no plano para o modelo antigo no T1:
  não são comparáveis (validação aqui tem ISIC 2020/MILK10k, mais difíceis, e 8 classes; o número
  antigo é no T1 em 7 classes). A comparação justa com o modelo antigo só sai no teste.

---

## 12/11 22:00–22:10 — Busca do E0: dois ajustes de infraestrutura antes de valer

**1. Memória virtual esgotada (erro 1455 do Windows).** A primeira tentativa da busca morreu com
"Couldn't open shared file mapping … error code 1455" (arquivo de paginação cheio). Causa: no
Windows cada processo de carregamento importa o torch inteiro (~2 GB de memória comprometida cada);
com 16 processos no treino + 16 na validação, mais um teste de diagnóstico que eu rodei em paralelo,
o sistema chegou a ~67 de 74 GB comprometidos. Isso também explicava os ~3 min de partida de cada
rodada (abrir 32 processos pesados).
- Mudança: **12 processos no treino, 4 na validação** (só inferência, leve), fila de pré-carga 2
  lotes por processo (era 4). Não muda nenhum resultado, só a velocidade e a memória.
- A tentativa que falhou ficou registrada em `experimentos.csv` com status `erro: RuntimeError`
  (falha de infraestrutura, não da configuração). O estudo do Optuna foi recriado do zero; com a
  mesma semente, o sorteio recomeça igual.

**2. Corte antecipado passa a valer só a partir da 4ª época.** Ao ver a primeira tentativa (cabeça
treinando sozinha nas 2 primeiras épocas, acurácia balanceada 0,30 na época 1), percebi que o corte
configurado a partir da 3ª época compararia essas tentativas com as outras logo depois de
descongelar a rede — penalizando aquecimento longo e cabeça congelada, que podem ser bons no final.
- Mudança: `MedianPruner(n_startup_trials=5, n_warmup_steps=3)` — as 5 primeiras tentativas nunca são
  cortadas; nas demais, o corte só olha da 4ª época em diante, comparando com a mediana das
  anteriores na mesma época. O plano fala em cortar "depois de 2 ou 3 épocas": fica em 3 completas.

---

## 12/11 22:15 — Plano da noite

Pedido: iterar E0, E1 e E2 até a manhã; se terminar antes, seguir para E3, E4 e E5.
- Fila 1 (rodando): E0_busca → E1 → E2, 30 tentativas cada, cada uma seguida do retreino da
  vencedora com o modelo salvo (`*_final_s42.pt`).
- Fila 2 (`F:\dermato\continuar_ate_E5.ps1`, agendada): espera a fila 1 acabar e roda vetores do
  PanDerm + teste rápido → E3 (fusão ConvNeXt+Swin, ramo ConvNeXt partindo do E1 final) → E4
  (+ PanDerm) → E5 (ensemble). Para antes dos finalistas e não abre os testes.
- Estimativa: E0_busca ~3–4 h, E1 ~8 h, E2 ~8 h. É provável que a manhã chegue durante a E1 ou a E2.

---

## 12/11 22:23 — Limite de tempo por busca em vez de 30 tentativas

A tentativa 0 da busca do E0 levou **15,9 min** (288 px, augmentation forte, 12 épocas) e chegou a
acurácia balanceada **0,701** (E0 receita: 0,696), mas com especificidade a 95% menor (0,635 × 0,689)
e ECE melhor (0,035 × 0,056). Nesse ritmo, 30 tentativas só do E0 seriam ~6–7 h e a E1/E2 (o foco)
mal começariam até a manhã.

Decisão: **orçamento de horas por etapa** (opção `--horas` da busca; a tentativa em andamento sempre
termina), priorizando as redes novas:

| Etapa | Horas de busca | Por quê |
|---|---|---|
| E0_busca | 1,25 (+ ~20 min já gastos) | o E0 é a régua; a busca é secundária |
| E1 ConvNeXt | 3,5 | etapa principal |
| E2 MaxViT | 3,5 | etapa principal |
| E3, E4 (se sobrar tempo) | 3 cada | fila 2, depois da E2 |

Cada busca termina com o retreino da vencedora (~15–25 min). O corte antecipado (mediana) deve
reduzir o custo médio por tentativa a partir da 6ª. Menos tentativas que as 30 do plano: o número
real de cada etapa fica registrado no `optuna.db` e será citado no notebook. A tentativa 1,
interrompida pela troca, foi marcada como falha no estudo.

---

## 12/11 22:40 — Tentativa repetida = medida do ruído entre rodadas

A tentativa 2 da E0_busca saiu com a **mesma configuração** da tentativa 0: ao retomar o estudo num
processo novo, o TPE com semente 42 recomeçou a sequência de sorteios. Resultado: 0,7007 × 0,6988 de
acurácia balanceada (spec95 0,635 × 0,637; AUC mel 0,890 × 0,890). Custou 16 min, mas dá a primeira
estimativa do **ruído entre rodadas de uma mesma configuração: ~0,002** (não-determinismo da GPU com
cudnn.benchmark e ordem dos processos de carregamento). Diferenças desse tamanho entre tentativas
são empate.
- Mudança (vale a partir da E1): semente do sorteador = 42 + nº de tentativas já no estudo.

---

## 12/11 23:40 — E0_busca, balanço das 10 primeiras tentativas

| # | val bal_acc | res | lr | lote | épocas | perda | aug | mixup | SoG | meta | nevos | balanc. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | **0,7345** | 224 | 4e-4 | 16 | 8 | ce | leve | 0,05 | não | sim | sem limite | não |
| 7 | 0,7222 | 288 | 1e-4 | 16 | 7 | ce | leve | 0,05 | sim | sim | 3.000 | sim |
| 0 | 0,7006 | 288 | 1e-4 | 64 | 12 | ce_ponderada | forte | 0,26 | sim | sim | 6.000 | sim |
| 4 | 0,6862 | 288 | <3e-5 | 16 | 13 | focal | leve | 0,16 | sim | não | 6.000 | não |
| 3 | 0,5155 | 224 | <3e-5 | 64 | 6 | ce_ponderada | forte | 0,39 | não | sim | 6.000 | não |

(6 e 8 cortadas na 4ª época; 1 falhou na troca de fila.)

O que já aparece:
- **lr baixo (< 3e-5) é ruim** em qualquer combinação: com 6–13 épocas a rede não sai do lugar.
- **Lote 16 nas duas melhores**: mais passos de atualização por época; com o B0 isso ajuda mais do que
  a estabilidade de lotes grandes.
- **Augmentation leve e pouco mixup** ganham da forte: a variação real entre ISIC 2019/2020/MILK10k
  já é grande; regularização demais atrapalha num modelo pequeno.
- **Suavização de rótulos 0,11–0,16** nas melhores (calibração melhor: ECE 0,055 na #9).
- A melhor (#9, 0,7345) bate o E0 receita (0,696) em +3,9 pontos — com a mesma rede. Parte grande do
  ganho que as próximas etapas mostrarem pode vir de parâmetros, não de arquitetura: é por isso que
  a comparação justa da E1/E2 é contra a E0_busca, não só contra a receita.
- 384 px ainda não foi sorteado.

---

## 12/11 23:59 — E0_busca concluída

11 tentativas em 96 min (8 completas, 2 cortadas, 1 falha de infraestrutura). Vencedora: **#9**.

Configuração vencedora: 224 px, lr 3,8e-4, weight decay 0,007, lote 16, 8 épocas, 1 época de
aquecimento, entropia cruzada simples (+ ajuste de prior, tau 0,75), suavização 0,11, augmentation
leve, cutmix com prob. 0,05, **sem** Shades of Gray, dropout 0,44, drop path 0,15, BatchNorm
congelada, 1 época só com a cabeça, metadados, **sem limite de nevos**, sem amostragem balanceada.

Retreino da vencedora (`E0_busca_final_s42`, modelo salvo): **reproduziu exatamente** a tentativa
(mesma semente, mesma configuração).

| Modelo (validação, TTA 8) | bal_acc | spec95 | AUC mel | AUC mal | ECE |
|---|---|---|---|---|---|
| E0 receita | 0,696 | 0,689 | 0,912 | 0,933 | 0,056 |
| **E0_busca** | **0,7345** | **0,703** | **0,917** | **0,942** | 0,055 |

Leitura:
- **+3,9 pontos de acurácia balanceada só ajustando parâmetros**, com a mesma rede e resolução.
- Diferenças em relação à receita que provavelmente explicam o ganho: lr ~4× maior (3,8e-4 × 1e-4),
  suavização de rótulos, dropout/drop path maiores, e usar todos os nevos (mais dados).
- Surpresa: **sem Shades of Gray** venceu. Com 11 tentativas não dá para afirmar que SoG atrapalha
  (a 2ª melhor usa SoG); fica como pergunta para E1/E2.
- Importância dos parâmetros (fANOVA do Optuna, só 10 tentativas, pouco confiável): dropout 0,27,
  lr 0,16, drop path 0,15, mixup 0,14, weight decay 0,13, suavização 0,10.

Próximo: E1 (ConvNeXt-Tiny, 224/288/384 px), 3,5 h de busca.

---

## 13/11 00:16 — E1, primeira tentativa e semeadura da busca

Tentativa 0 da E1 (ConvNeXt-Tiny com a mesma configuração sorteada na tentativa 0 do E0: 288 px,
lote 64, perda ponderada, augmentation forte): **bal_acc 0,732, spec95 0,729, AUC mel 0,925, ECE 0,018**
(16 min). A mesma configuração no B0 deu 0,701/0,635: só trocar a rede já dá +3 pontos e +9 pontos de
especificidade a 95%.

Decisão: com ~3,5 h (~12 tentativas), **enfileirei no estudo da E1 a configuração vencedora da
E0_busca com resolução 384 px** (`enqueue_trial`). Motivo: começar pela região que já se mostrou boa
em vez de depender só de sorteios, e testar de cara a hipótese da E1 (mais resolução preserva a rede
de pigmento e as bordas → melanoma). O restante das tentativas segue o TPE normalmente; a tentativa
enfileirada aparece no `optuna.db` como qualquer outra.

---

## 13/11 02:20 — E1, balanço das 11 primeiras tentativas

| # | val bal_acc | spec95 | AUC mel | ECE | res | lr | lote | épocas | perda | aug | mix | SoG |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | **0,769** | **0,773** | **0,933** | 0,314 | 384 | 2,6e-4 | 16 | 9 | focal | leve | 0,26 | não |
| 2 (semeada) | 0,734 | 0,749 | 0,931 | 0,048 | 384 | 3,8e-4 | 16 | 8 | ce | leve | 0,05 | não |
| 0 | 0,732 | 0,729 | 0,925 | 0,018 | 288 | 1,0e-4 | 64 | 12 | ponderada | forte | 0,26 | sim |
| 6 | 0,726 | 0,712 | 0,920 | 0,065 | 288 | 1,5e-4 | 16 | 7 | ce | leve | 0,05 | sim |

(5, 7, 8, 9 cortadas na 4ª época; 1 e 4 fracas, lr baixo ou augmentation média com mixup 0,40.)

O que aprendi:
- **Resolução é o fator mais forte da E1 — hipótese do plano confirmada.** Prova quase controlada:
  a tentativa 8 sorteou exatamente a configuração da 2, mas a **224 px**: estava em 0,642 na época 4 e
  foi cortada; a 2, a **384 px**, chegou a 0,734. As três melhores são 384 ou 288.
- A semeadura com a vencedora do E0 (tentativa 2) já deu um ponto de partida bom (0,734, spec95
  0,749) e o TPE refinou a partir dela: a #10 mantém lote 16, augmentation leve, sem SoG, metadados,
  todos os nevos, BN congelada, 1 época de cabeça — e troca a perda por **focal** com mais mixup/cutmix.
- **ECE alto na #10 (0,31)**: focal + suavização de rótulos produzem probabilidades "achatadas" (o
  modelo fica pouco confiante de propósito). As decisões (argmax com ajuste de prior, limiar de
  triagem) não dependem da escala, por isso bal_acc e spec95 são as melhores; a calibração por
  temperatura (E5) corrige a escala sem mexer no ranking. Registro aqui para não esquecer de reportar
  o ECE calibrado e o bruto.
- Sem Shades of Gray de novo nas duas melhores (E0 e agora E1).

---

## 13/11 04:00 — E1 concluída

16 tentativas em 3,5 h (9 completas, 7 cortadas). Vencedora: **#10**, retreinada como `E1_final_s42`
(reproduziu exatamente: 0,7692).

Configuração: ConvNeXt-Tiny (ImageNet-22k), **384 px**, lr 2,6e-4, weight decay 0,017, lote 16,
9 épocas, sem aquecimento, **focal (gama 2,97)**, suavização 0,16, augmentation leve, mixup com
prob. 0,26, sem Shades of Gray, dropout 0,27, drop path 0,06, 1 época só com a cabeça, metadados,
todos os nevos, sem amostragem balanceada.

| Modelo (validação, TTA 8) | bal_acc | spec95 | AUC mel | AUC mal | ECE |
|---|---|---|---|---|---|
| E0 receita | 0,696 | 0,689 | 0,912 | 0,933 | 0,056 |
| E0_busca | 0,7345 | 0,703 | 0,917 | 0,942 | 0,055 |
| **E1** | **0,769** | **0,773** | **0,933** | **0,951** | 0,314 |

Leitura:
- E1 × E0_busca (ambos otimizados): **+3,5 pontos de acurácia balanceada e +7 pontos de
  especificidade a 95%** — com a sensibilidade fixa em 95%, ~7% a mais dos benignos sairiam da fila
  prioritária. Esse é o ganho atribuível à rede + resolução, já descontado o ganho de parâmetros.
- Importância (fANOVA, 16 tentativas): gama da focal 0,31, dropout 0,14, mixup 0,12, lr 0,11.
  **Gama ficou no limite superior da faixa (2,97 de 0,5–3)**: a faixa pode estar cortando o ótimo.
  Anoto como possível extensão (gama até 5) se houver nova rodada da E1.
- ECE bruto alto (0,31) pelos motivos já anotados (focal + suavização); calibrar na E5.

## 13/11 04:05 — E2 semeada com a vencedora da E1

Mesma estratégia da E1: enfileirei no estudo da E2 a configuração vencedora da E1, sem a resolução
(E2 é fixa em 224 px) e com decaimento por camada 0,8 (o meio da faixa 0,6–1,0; parâmetro que só
existe na E2). A primeira tentativa (#0) já tinha sido sorteada; a semeada é a #1.

---

## 13/11 05:45 — E2: memória da GPU estourada e correção

Primeiras tentativas da E2 (MaxViT-Tiny, 224 px):

| # | val bal_acc | spec95 | AUC mel | ECE | observação |
|---|---|---|---|---|---|
| 1 (semeada com a E1) | **0,749** | 0,727 | 0,925 | 0,323 | 28 min |
| 0 | 0,693 | 0,693 | 0,917 | 0,025 | |
| 2 | 0,592 | 0,635 | 0,873 | 0,079 | lr 1,3e-5, 6 épocas |

**Problema:** a tentativa 3 (lote 64, BatchNorm treinável) levou **25 min só na primeira época**
(esperado ~2 min). `nvidia-smi`: 11,8 de 12,2 GB ocupados com GPU a 100%. Eu tinha deixado a MaxViT
fazer passos de até 64 imagens (o teste de velocidade foi com 32 e BN congelada); com BN treinável
passou de 12 GB e o driver do Windows começou a usar a RAM do sistema como extensão da VRAM —
sem erro, só ~10× mais lento. As 13 épocas teriam consumido ~5 h, o orçamento inteiro.

Correções:
- Passo máximo da MaxViT: **32 imagens** (lotes maiores viram acumulação de gradiente: mesmo
  resultado matemático, BatchNorm à parte). Fusão (E3/E4): 48 em vez de 64, por precaução.
- **Teto de 90% da VRAM** no PyTorch (`set_per_process_memory_fraction`): se algo não couber, dá
  erro de memória na hora (a tentativa falha e a busca segue) em vez de rodar lento escondido.
- Tentativa 3 marcada como falha no estudo (problema de infraestrutura, não da configuração).
- A E2 perdeu ~45 min com isso: recebeu **2 h a partir de agora** (termina ~07:50, mais o retreino).

Leitura parcial: a MaxViT a 224 px com a configuração da E1 (0,749) supera o B0 otimizado a 224 px
(0,7345) mas fica abaixo do ConvNeXt a 384 px (0,769). Para separar "arquitetura" de "resolução", a
comparação justa no notebook deve citar que a E2 ficou em 224 px por custo (4 min/época a 384).

**06:45 — refinamento dirigido na E2.** Depois da correção, as tentativas 4 (0,732) e 5 (0,662)
completaram e 6, 7, 8 foram cortadas na 4ª época. Com ~1 h restante (~2 tentativas de 28 min),
sorteios novos têm pouca chance de achar algo; enfileirei **duas variações da melhor (#1)** mudando
só o decaimento por camada (parâmetro exclusivo da E2): 0,65 e 0,95 (a #1 usou 0,8). Responde se o
decaimento importa e refina ao redor da melhor.

---

## 13/11 07:48 — E2 encerrada (busca)

12 tentativas (busca de 3,5 h + 2 h, com ~45 min perdidos no estouro de memória): vencedora **#1**
(a semeada com a configuração da E1, decaimento por camada 0,8): **bal_acc 0,749, spec95 0,727,
AUC mel 0,925**. Retreino como `E2_final_s42` em andamento.

Teste do decaimento por camada (mesma configuração da #1, só ele muda):

| decaimento | bal_acc | spec95 | AUC mel |
|---|---|---|---|
| 0,65 (#10) | 0,746 | **0,754** | 0,921 |
| 0,80 (#1) | **0,749** | 0,727 | 0,925 |
| 0,95 (#11) | 0,745 | 0,747 | 0,928 |

As três acurácias estão dentro do ruído entre rodadas (~0,002–0,004): **o decaimento por camada não
muda o resultado de forma mensurável** nesta configuração. A especificidade a 95% varia mais
(0,727–0,754) e não é monotônica, o que sugere ruído também.

Comparação das três etapas (validação, TTA 8):

| Modelo | res | bal_acc | spec95 | AUC mel | AUC mal |
|---|---|---|---|---|---|
| E0 receita (B0) | 224 | 0,696 | 0,689 | 0,912 | 0,933 |
| E0_busca (B0) | 224 | 0,7345 | 0,703 | 0,917 | 0,942 |
| E2 (MaxViT) | 224 | 0,749 | 0,727 | 0,925 | 0,945 |
| **E1 (ConvNeXt)** | **384** | **0,769** | **0,773** | **0,933** | **0,951** |

Leitura:
- A 224 px, a MaxViT (convolução + atenção) ganha do B0 otimizado: +1,5 ponto de bal_acc, +2,4 de
  spec95. Pequeno, mas na direção da hipótese da E2.
- O ConvNeXt a 384 px é o melhor em tudo. Como a E1 a 224 px foi cortada (0,642 na época 4) e a E2
  ficou só em 224 por custo, **não dá para dizer que a MaxViT é pior que o ConvNeXt**: o que se sabe é
  que resolução pesa mais que a troca de arquitetura. Para fechar essa pergunta seria preciso a E2 a
  384 px (~1 h por tentativa) ou o ConvNeXt completo a 224 px.
- A hipótese específica da E2 (menos confusão entre ceratose benigna, ceratose actínica e carcinoma
  espinocelular) precisa da matriz de confusão por classe — vai para a análise no notebook, com as
  previsões salvas em `resultados/previsoes/`.

---

## 13/11 08:17 — E2 retreinada; teste das hipóteses por classe

`E2_final_s42`: bal_acc **0,746**, spec95 0,709, AUC mel 0,922, AUC mal 0,943 (a tentativa #1 deu
0,749 / 0,727). Diferente de E0 e E1, a MaxViT **não reproduziu exatamente** com a mesma semente
(provável não-determinismo dos kernels de atenção). Medida de ruído útil: entre duas rodadas da mesma
configuração, **bal_acc varia ~0,3 ponto e spec95 ~2 pontos**. Diferenças de spec95 abaixo de ~2
pontos entre modelos devem ser tratadas como empate.

Acerto por classe na validação (modelos finais, TTA, tau da validação):

| Modelo | ak | bcc | bkl | df | mel | nv | scc | vasc | confusão bkl/ak/scc |
|---|---|---|---|---|---|---|---|---|---|
| E0 receita | 0,616 | 0,660 | 0,558 | 0,787 | 0,625 | 0,794 | 0,609 | 0,92 | 0,191 |
| E0_busca | 0,627 | 0,766 | 0,564 | 0,872 | 0,615 | 0,833 | 0,639 | 0,96 | 0,179 |
| E2 MaxViT | 0,593 | 0,777 | **0,708** | 0,787 | 0,634 | **0,869** | 0,658 | 0,94 | 0,180 |
| **E1 ConvNeXt** | 0,576 | **0,807** | 0,658 | **0,915** | **0,770** | 0,803 | **0,663** | 0,96 | **0,167** |

(confusão bkl/ak/scc = fração das imagens dessas três classes que foram para uma das outras duas.
n na validação: ak 177, bcc 945, bkl 541, df 47, mel 919, nv 2.940, scc 202, vasc 50.)

- **Hipótese da E1 (melanoma) confirmada com folga**: acerto em melanoma 0,62 → **0,77** (+14 pontos
  sobre o E0_busca, com 919 melanomas na validação). É o maior ganho isolado da noite e o que mais
  importa clinicamente.
- **Hipótese da E2 (menos confusão bkl/ak/scc) não confirmada**: a MaxViT acerta muito mais ceratose
  benigna (0,708, melhor de todos), mas a confusão entre as três fica igual à do B0 otimizado (0,180 ×
  0,179) e o acerto em ceratose actínica cai. Quem menos confunde é a E1 (0,167). Resultado negativo
  registrado, como manda a regra da evolução.
- AK é a classe mais difícil para todos (0,58–0,63) e a única em que nenhuma etapa melhorou: candidata
  óbvia para a E4 (PanDerm, classes raras) e para o ensemble.
- Ponto de partida para a E3 (fusão): E1 e E2 erram em classes diferentes (E2 melhor em bkl/nv, E1 em
  mel/bcc/df) — sinal favorável à hipótese da fusão/ensemble de que olhares diferentes se complementam.

Fila 1 (E0 → E0_busca → E1 → E2) terminou às 08:17. Fila 2 começou: vetores PanDerm + teste rápido
→ E3 (3 h) → E4 (3 h) → E5.

---

## 13/11 08:20–08:30 — Vetores do PanDerm e teste rápido

Dois bugs meus no `panderm.py`, que nunca tinha rodado de verdade: (1) a classe de dataset definida
dentro da função não pode ser enviada aos processos de carregamento no Windows (modo spawn) — movida
para o nível do módulo; (2) os IDs das imagens foram salvos como texto (array de objetos) e o
`np.load` precisa de `allow_pickle=True` para lê-los. Nenhum resultado foi afetado.

Extração: 42.652 vetores de 1024 números em 2,6 min (ViT-L/16, 224 px, sem ajuste fino).

Teste rápido (treino → validação, vetores + 14 metadados, padronizados; pesos balanceados por classe):

| Classificador | bal_acc | spec95 | AUC mel | AUC mal | tempo |
|---|---|---|---|---|---|
| regressão logística | **0,632** | 0,613 | 0,897 | 0,919 | 0,3 min |
| MLP (512) | 0,621 | 0,653 | 0,910 | 0,921 | 3,5 min |
| LightGBM | 0,541 | **0,674** | **0,911** | **0,935** | 1,5 min |
| (E0 receita, para comparar) | 0,696 | 0,689 | 0,912 | 0,933 | 9,5 min de GPU |

Leitura:
- **As AUCs do PanDerm congelado empatam com o E0 receita** (AUC maligno × benigno 0,935 × 0,933;
  melanoma 0,911 × 0,912) sem nenhum ajuste fino, em 1–4 min de CPU. É o "achado" previsto no plano:
  um modelo pronto chega perto de uma CNN treinada, a uma fração do custo — para a decisão binária.
- Na classificação em 8 classes (bal_acc) fica 6–15 pontos abaixo: os vetores separam bem
  maligno/benigno, mas classificadores simples erram mais nas classes raras.
- Decisão: **a E4 segue como planejado** (vetor congelado como terceiro ramo da fusão). Os vetores
  não são fracos; o valor esperado está justamente em complementar a CNN nas classes raras.

## 13/11 08:30 — E3 (fusão ConvNeXt + Swin) iniciada e semeada

Busca de 3 h. O ramo ConvNeXt parte dos pesos de `E1_final_s42`; o Swin-Tiny recebe a imagem
redimensionada para 224 px (a janela 7 do Swin não divide 288/384). Semeada com a configuração
vencedora da E1 + junção por concatenação (a do diagrama do plano), taxa do ramo Swin 0,5× a do
ConvNeXt, 1 época com os dois ramos congelados (só junção e cabeça aprendem) e cabeça de 512.

---

## 13/11 ~09:00 — E3 interrompida; rodada extra de E1 e E2

Pedido: parar a E3 e otimizar mais a E1 e a E2 antes de seguir. A E3 não tinha nenhuma tentativa
completa; o estudo foi apagado (será refeito partindo da melhor E1).

**Por que ainda há espaço** (raciocínio):
- E1: só 16 tentativas (9 completas), a melhor veio na 10ª; o gama da focal — parâmetro mais
  importante — ficou no teto da faixa (2,97 de 3).
- E2: só 7 tentativas completas e presa em 224 px. Como na E1 a resolução foi o fator mais forte,
  hoje não dá para separar "MaxViT é pior" de "224 px é pior".

**E1r — E1 refinada (4 h).** Espaço estreitado com o que a noite mostrou (`refinados()` em busca.py):
384 ou **448 px** (limite do cache), lr 1,5e-4–4e-4, focal com **gama 2–5**, lote 16/32, 8–15
épocas, suavização 0,05–0,2, augmentation leve/média, mixup/cutmix até 0,4, dropout 0,1–0,5, drop
path 0–0,2, todos os nevos ou 6.000; fixos: sem SoG, BN congelada, metadados, sem amostragem
balanceada (nenhuma das melhores usou o contrário). Corte antecipado a partir de 3 tentativas
completas (antes 5), porque a busca começa semeada.
- Semeadura: (a) a vencedora da E1 **registrada com o resultado conhecido** (0,7692) sem retreinar —
  o ConvNeXt é determinístico (o retreino reproduziu exatamente), então repetir custaria 27 min sem
  informação nova; (b) a mesma a **448 px**; (c) a mesma com **gama 4,5**. (b) e (c) testam
  diretamente as duas hipóteses de melhoria.

**E2_384 — MaxViT a 384 px (6 h).** `maxvit_tiny_tf_384`, mesmo espaço estreitado com 6–12 épocas
(cada época ~4 min), decaimento por camada fixo em 0,8 (não mudou nada na E2). Semeada com a
vencedora da E2 (= configuração da E1). Objetivo: comparação com a E1 na mesma resolução.

**Depois:** a E3 parte da **melhor** E1 (E1 ou E1r, pela validação), e a E5 junta a melhor de cada
família (E1/E1r, E2/E2_384, E3, E4). Fila: `--ate E5 --horas E1r=4,E2_384=6,E3=3,E4=3`.

**09:17 — E1r, hipótese 1 (448 px) descartada.** Vencedora da E1 a 448 px: bal_acc **0,7575**,
spec95 0,750, AUC mel 0,931 (36,8 min) × a mesma a 384 px: 0,7692 / 0,773 / 0,933 (26,6 min). Época a
época ficou ~2 pontos atrás o tempo todo (época 8 sem TTA: 0,741 × 0,762). Mais resolução além de
384 px não ajuda nesta configuração — possivelmente porque o cache tem menor lado 448, então a 448 px
o recorte aleatório precisa ampliar a imagem (perde o efeito de "zoom" da augmentation) e a lesão
ocupa uma fração parecida da imagem de 384. Fica 384 px como resolução da família ConvNeXt.

**09:46 — E1r, hipótese 2 (gama além de 3) confirmada, com ganho pequeno.** Vencedora da E1 com
gama 4,5 (só ele muda): bal_acc **0,7753** (+0,6 ponto sobre 0,7692), spec95 **0,780** (+0,7), AUC mel
0,934, AUC mal 0,952; ECE pior (0,387 × 0,314: focal mais forte achata ainda mais as probabilidades —
a calibração da E5 cuida disso). O ganho de acurácia passa do ruído medido entre rodadas (~0,3); o de
spec95 fica dentro do ruído (~2), então conto como "não piora". Interpretação: gama maior concentra a
perda nos exemplos difíceis — coerente com o problema ter muitas lesões fáceis (nevos) e classes
raras difíceis. O TPE agora explora gama 2–5 no resto da busca.

---

## 13/11 10:43 — Fila morta junto com a sessão do assistente; relançada

A fila parou por volta das 10:03, no meio da tentativa 4 da E1r (época 5/8), **sem erro no log**:
os processos iniciados pela sessão do Claude Code estavam presos a ela no Windows e foram encerrados
quando a sessão terminou. Perda: ~40 min de GPU parada + a tentativa 4 (marcada como falha no estudo).
Tudo que já tinha terminado estava salvo (tentativas 1–3, `optuna.db`, `experimentos.csv`).
- Correção: a fila agora é criada via WMI (`Win32_Process.Create`), fora da árvore de processos da
  sessão — sobrevive se o assistente fechar.
- Orçamento: a E1r já tinha usado ~1,5 h das 4 h; recebeu mais **2,5 h** (até ~13:15).

Estado da E1r até aqui: 448 px 0,7575; gama 4,5 a 384 px **0,7753** (melhor); E1 original
(registrada) 0,7692.

**10:48 — Prioridade para a E2 a 384 px.** A pedido, a busca da E1r foi encerrada com o ganho já
obtido (melhor: gama 4,5, bal_acc 0,7753); a tentativa 5, recém-iniciada, foi marcada como falha.
Nova ordem (lançada via WMI): **E2_384 por 6 h** (até ~16:50, mais o retreino) → retreino da melhor
E1r (`--so-final`, salva `E1r_final_s42.pt`) → fila E3 → E4 → E5 (a E3 parte da melhor E1).

---

## 13/11 11:20 — Modelo utilizável fora desta máquina

Pedido: deixar o modelo usável por quem clonar o repositório (ex.: Felipe).
- `dermato/prever.py`: roda um modelo final em imagens/pastas novas e grava CSV com classe,
  probabilidades, escore de malignidade e triagem (tau e limiar da validação). Reproduz o
  pré-processamento do cache do treino (draft do JPEG + LANCZOS + regravação JPEG 95 + torchvision).
- Verificação em 6 imagens da **validação** (uma por classe; nada do teste): acertou as 6 com E1 e E0.
  Probabilidades × as salvas no treino: com os metadados reais, diferença máxima **0,009** (float32 na
  CPU) e **0,002** em bf16 → o script reproduz o treino. Minha primeira hipótese para uma diferença de
  0,06 (precisão bf16 × float32) estava errada: a causa era rodar sem metadados. Por isso o README
  recomenda passar `--meta`.
- Caminhos portáveis: `config.achar()` acha pesos/calibrações pelo nome em `DERMATO_RESULTADOS` ou em
  `resultados/` do repositório (antes o `testar.py` dependia do caminho absoluto deste PC).
- `requirements.txt`, `scripts_dermato/baixar_dados.py` (versão em Python do download, retomável) e
  seção nova no README.
- Pesos (23–107 MB; o da E1 passa do limite de 100 MB do GitHub) ficam fora do git
  (`resultados/modelos/` no .gitignore); somas SHA-256 em `resultados/modelos_sha256.txt`. Cópias
  prontas para publicar em `F:\dermato\pesos_para_compartilhar\`.

---

## 13/11 11:45 — E2 a 384 px: VRAM transbordando de novo; teto mais baixo

A primeira tentativa da E2_384 fez a época 2 em 6,6 min, mas as épocas 3 e 4 levaram **~21 min cada**.
Primeiro atribuí a lentidão aos meus testes do `prever.py` na CPU (que de fato coincidiram com a
época 3), mas a época 4 também foi lenta depois deles. Diagnóstico com `nvidia-smi`: **11,9 de 12,2 GB**
de VRAM e só **109 W** de consumo com "97% de uso" (a GPU trabalhando de verdade passa de 200 W) →
o driver do Windows estava usando a RAM do sistema como extensão da VRAM, de novo.
- Causa: o teto de 90% vale só para o PyTorch, mas este PC já ocupa **~1,4 GB de VRAM com os
  programas abertos** (wallpaper animado, Discord, iCUE…) + o contexto da CUDA. A época 2 foi rápida
  porque o cache do PyTorch ainda não tinha crescido até o teto.
- Também: só 3,7 GB de memória virtual comprometida livre (perto do erro 1455 de ontem).
- Correções: **teto de VRAM 80%** (~9,8 GB; esta configuração mede ~8 GB) e **10 processos de
  carregamento** em vez de 12 (a MaxViT a 384 px é limitada pela GPU, ~94 img/s).
- A tentativa (semente, época 4/9, ainda em 0,697 sem TTA) foi interrompida — terminar a 21 min/época
  custaria ~1h45, recomeçar ~1 h — marcada como falha e reenfileirada com a mesma configuração.
  Orçamento da E2_384: 5 h a partir de agora (já tinha perdido ~1 h).
- Lição registrada: não rodar testes pesados na CPU durante treinos (no `prever.py`, a CPU usa
  todos os núcleos e tira processador dos processos de carregamento).

---

## 13/11 12:50 — E2 a 384 px: arquitetura × resolução (comparação controlada)

Primeira tentativa da E2_384 (semente = mesma configuração da vencedora da E1/E2), 60 min, épocas
estáveis em 6,5 min depois do teto de VRAM de 80% (consumo da GPU 220 W × 109 W antes):
**bal_acc 0,7565, spec95 0,774, AUC mel 0,928, AUC mal 0,950, ECE 0,310**.

Mesma configuração, só muda rede e/ou resolução (validação, TTA 8):

| Modelo | bal_acc | spec95 | mel | scc | df | bkl | nv | conf. bkl/ak/scc |
|---|---|---|---|---|---|---|---|---|
| E2 MaxViT 224 px (final) | 0,746 | 0,709 | 0,634 | 0,658 | 0,787 | 0,708 | 0,869 | 0,180 |
| E2 MaxViT **384 px** | 0,7565 | **0,774** | 0,737 | **0,708** | 0,809 | 0,643 | 0,847 | 0,179 |
| E1 ConvNeXt 384 px (final) | **0,769** | 0,773 | **0,770** | 0,663 | **0,915** | 0,658 | 0,803 | **0,167** |

Conclusões:
1. **Resolução explica a maior parte da distância entre E1 e E2.** Na própria MaxViT, 224 → 384 px:
   +10 pontos em melanoma, +6,5 em spec95, +5 em scc. É o mesmo efeito visto na E1.
2. **Na mesma resolução, as arquiteturas empatam na triagem** (spec95 0,774 × 0,773, dentro do ruído de
   ~2 pontos). ConvNeXt à frente em bal_acc (+1,2), melanoma (+3) e dermatofibroma (+11, mas só 47
   imagens: ±5 pontos de incerteza); MaxViT melhor em carcinoma espinocelular (+4,5).
3. **Hipótese da E2 segue não confirmada**: a confusão bkl/ak/scc não muda com a resolução (0,180 →
   0,179); atenção em cada estágio não resolveu essa confusão nesta base.
4. Erram em classes diferentes (MaxViT: scc, nv; ConvNeXt: mel, df) → argumento a favor do ensemble E5.

---

## 13/11 16:55 — E2 a 384 px: busca encerrada

7 tentativas (5 completas, 1 cortada, 1 falha da VRAM) em ~5 h; ~55 min por tentativa completa.

| # | bal_acc | spec95 | lr | lote | épocas | gama | aug | mixup | nevos |
|---|---|---|---|---|---|---|---|---|---|
| **3** | **0,768** | 0,735 | 2,8e-4 | 16 | 9 | 2,8 | leve | 0,23 | 6.000 |
| 6 | 0,763 | 0,766 | 1,6e-4 | 32 | 10 | 4,8 | leve | 0,20 | todos |
| 1 (semente) | 0,756 | **0,774** | 2,6e-4 | 16 | 9 | 3,0 | leve | 0,26 | todos |
| 2 | 0,756 | 0,764 | 3,4e-4 | 16 | 8 | 3,2 | média | 0,38 | todos |
| 5 | 0,747 | 0,725 | 3,9e-4 | 16 | 6 | 2,9 | média | 0,04 | 6.000 |

Leitura:
- Vencedora pela acurácia balanceada: **#3 (0,768)**, praticamente empatada com a E1 (0,769) — mas
  com spec95 menor (0,735). As cinco completas ficam entre 0,747 e 0,768: com 384 px, a MaxViT é
  pouco sensível aos parâmetros dentro desta faixa estreitada.
- **Conflito de critério**: a melhor bal_acc (#3) não é a melhor spec95 (#1, 0,774). O protocolo usa
  bal_acc para escolher; registro aqui que, para o uso de triagem (especificidade a 95% de
  sensibilidade), a #1 seria a escolha. Vale reportar as duas no notebook.
- Gama alto (4,8, #6) não repetiu na MaxViT o ganho que deu no ConvNeXt (0,763 × 0,768 da #3 com 2,8).
- Limitar nevos a 6.000 apareceu nas duas extremidades (#3 melhor, #5 pior): sem efeito claro.

Retreino da #3 como `E2_384_final_s42` em andamento; depois retreino da E1r (gama 4,5) e E3 → E4 → E5.

---

## 13/11 18:10 — E1r retreinada NÃO reproduziu: ruído maior do que eu estimava (correção)

`E1r_final_s42` (mesma configuração da tentativa #2: gama 4,5) deu **bal_acc 0,7646**, spec95 0,770,
AUC mel 0,933 — contra **0,7753** / 0,780 na tentativa. Até aqui o ConvNeXt reproduzia exatamente
(E1: 0,7692 nas duas vezes). Causa: às 11:45 reduzi os processos de carregamento de 12 para 10 (teto de
VRAM). Cada processo tem sua própria semente de augmentation; mudar o número de processos muda os
recortes/flips/mixup de cada imagem. Resultado: o mesmo modelo, com outra sequência aleatória de
augmentation, varia **~1 ponto de bal_acc**.

Correções no que registrei antes:
- **Ruído entre rodadas ≈ 1 ponto de bal_acc** (não ~0,3: aquela estimativa vinha de rodadas com a mesma
  sequência de augmentation, só com não-determinismo da GPU). Para spec95 continuo usando ~2 pontos.
- **O ganho do gama 4,5 (+0,6) está dentro do ruído**: não dá para afirmar que melhorou a E1. A
  conclusão correta é "gama 3–4,5 empatam".
- O mesmo vale para E1 × E2 a 384 px (0,769 × 0,768): empate.
- Isso reforça o passo do plano de rodar os finalistas com 3 sementes e comparar média ± desvio.

Efeito prático: a fila escolhe a base da E3 pela bal_acc dos modelos finais salvos → **E1 original
(0,7692)**, não a E1r (0,7646). Escolha razoável: as duas empatam.
**18:06 — E3 iniciada** partindo de `E1_final_s42`, semeada com a vencedora da E1 + junção por concatenação, Swin a 0,5× o lr do ConvNeXt, 1 época com os ramos congelados, cabeça de 512 (mesma semeadura planejada às 08:30).

---

## 13/11 21:10 — E3 (fusão ConvNeXt + Swin): busca encerrada

10 tentativas em 3 h (6 completas, 4 cortadas). O ramo ConvNeXt partiu de `E1_final_s42`.

| # | bal_acc | spec95 | res | junção | lr | Swin × lr | ramos congelados | cabeça | perda |
|---|---|---|---|---|---|---|---|---|---|
| **7** | **0,767** | 0,756 | 384 | **atenção cruzada** | 5,3e-5 | 0,78 | 1 época | 256 | focal |
| 0 | 0,757 | 0,721 | 288 | porta | 1,0e-4 | 0,20 | 0 | 256 | ponderada |
| 1 (semente) | 0,753 | **0,773** | 384 | concatenação | 2,6e-4 | 0,50 | 1 época | 512 | focal |
| 3 | 0,747 | 0,718 | 224 | porta | 1,3e-5 | 0,15 | 3 | 512 | focal |
| 4 | 0,746 | 0,722 | 224 | atenção | 2,4e-5 | 0,17 | 2 | 512 | focal |
| 2 | 0,707 | 0,717 | 288 | concatenação | 3,5e-4 | 0,59 | 0 | 256 | ce |

Leitura:
- **A fusão não superou a E1 sozinha**: melhor E3 0,767 × E1 0,769 (empate, dentro do ruído de ~1
  ponto); spec95 0,756–0,773 × 0,773. Hipótese da E3 ("dois olhares separados se complementam mais
  do que misturados") **não confirmada** nesta forma — o ramo Swin a 224 px não acrescentou informação
  que o ConvNeXt a 384 px já não tivesse.
- Padrão claro de lr: como o ramo ConvNeXt já parte treinado (E1), **lr baixo (5e-5)** funcionou melhor
  que o da E1 (2,6e-4, tentativa semeada): com lr alto o ConvNeXt "esquece" parte do que aprendeu.
  A vencedora usa o mesmo lr base para os dois ramos praticamente (Swin 0,78×).
- 384 px de novo nas melhores; 224 px sempre atrás (o ramo ConvNeXt perde a resolução que o fazia bom).
- Junção por atenção cruzada venceu, mas com só 1–2 tentativas por tipo de junção não dá para
  separar o efeito da junção do efeito do lr.
- Próximo: E4 adiciona o vetor do PanDerm como terceiro ramo (3 h).

---

## 13/11 21:45 — Notebook de registro criado

O plano pede um `notebook.ipynb` com uma seção por modelo; até aqui o registro estava no DIARIO.md,
em `experimentos.csv` e no `optuna.db`. Criado **`notebook_dermatoscopia.ipynb`** (o `notebook.ipynb`
antigo, da versão TensorFlow, foi mantido), gerado por `scripts_dermato/gerar_notebook.py` a partir dos
resultados salvos — não treina nada e roda sem imagens/GPU. Cada etapa tem as quatro partes do plano
(como funciona; o que foi variado, com faixas e nº de tentativas; resultados com as 5 melhores, a
vencedora, a evolução da busca e a importância dos parâmetros; o que funcionou e por quê) e a seção
final "linha de raciocínio" junta tudo (tabela e gráfico dos finais, acerto por classe, matrizes de
confusão E1 × E2 a 384 px). Funções de tabela/gráfico em `dermato/relatorio.py`; gráficos seguindo
o guia de visualização (paleta validada, um eixo, pontos em vez de barras com eixo cortado, legenda
fora da área de dados — conferidos visualmente depois de renderizar). Previsões de validação de todos
os modelos finais copiadas para `resultados/previsoes/`. Regenerar quando E4/E5 terminarem.

## 13/11 21:50 — Plano da noite (2ª noite)

Pedido: treinar os modelos que faltam, bem otimizados, e deixar um resumo pela manhã.
- Fila atual: retreino da E3 → **E4** (fusão + PanDerm, 3 h, semeada com a vencedora da E3) → **E5**
  (ensemble da melhor de cada família + calibração por temperatura).
- Depois (`F:\dermato\depois_finalistas.ps1`, via WMI): **finalistas** — os 3 melhores modelos pela
  validação, retreinados com mais 2 sementes (43, 44). Motivo: o ruído entre rodadas (~1 ponto) é do
  tamanho das diferenças entre E1, E2 a 384 px e E3; média ± desvio de 3 sementes é o que separa
  diferença real de ruído (passo previsto no plano antes de abrir os testes).
- Os testes T1/T2 continuam fechados.

**22:03 — E4 iniciada** (3 h): ramo ConvNeXt a partir de `E1_final_s42` + Swin + vetor congelado do PanDerm. Semeada com a vencedora da E3 (atenção cruzada, lr 5,3e-5, 384 px) + dropout 0,3 no vetor do PanDerm e vetor normalizado. Retreino da E3: bal_acc 0,7636, spec95 0,761 (tentativa: 0,767 — dentro do ruído).

---

## 14/11 02:25 — E4 (fusão + PanDerm) concluída

9 tentativas em 3 h (7 completas, 2 cortadas). Vencedora: **#1, a semeada** (configuração da E3 + vetor
PanDerm com dropout 0,3, normalizado): bal_acc 0,7656, spec95 0,756. Retreino `E4_final_s42`: **0,7567 /
0,749**, AUC mel 0,932 (dentro do ruído da tentativa).

- **O PanDerm não acrescentou nada à fusão**: a mesma configuração sem ele (E3 #7) deu 0,767 / 0,756 —
  idêntico. Hipótese da E4 ("um modelo que viu muito mais pele compensa a falta de exemplos raros")
  **não confirmada**: os vetores congelados repetem informação que a CNN treinada já tem.
- As melhores tentativas tiveram calibração boa (ECE 0,024–0,036 nas #3 e #8), sem a focal forte.

## 14/11 02:26 — E5 (ensemble) concluída

Membros: a melhor de cada família pelo critério da fila (E1, E2 a 384 px, E3, E4). Ajustado na metade A
da validação (pesos/LightGBM, temperatura, tau, limiar), avaliado na metade B (divisão por grupo).
Testados todos os subconjuntos × média, média ponderada e LightGBM.

| Na metade B | bal_acc | spec95 | AUC mel | AUC mal | ECE |
|---|---|---|---|---|---|
| E1 sozinha | 0,769 | 0,757 | 0,929 | 0,930 | 0,190 |
| E3 sozinha | 0,768 | 0,734 | 0,938 | 0,948 | 0,051 |
| **média dos 4 + temperatura (vencedor)** | **0,775** | **0,775** | **0,945** | **0,950** | **0,056** |
| média E1 + E2 a 384 px | 0,770 | **0,777** | 0,942 | 0,945 | 0,053 |

- bal_acc +0,6 sobre a E1: dentro do ruído. **Os ganhos reais são de ordenação e calibração**: AUC
  maligno 0,930 → 0,950, AUC melanoma 0,929 → 0,945, spec95 +1,8 ponto e ECE 0,19 → 0,056. Hipótese da E5
  ("modelos diferentes erram em imagens diferentes") **confirmada** nas métricas de triagem.
- Média simples ganhou de média ponderada e de LightGBM (com metade da validação para ajustar, os
  métodos com mais parâmetros não compensam).
- **Anomalia a investigar**: a E2 a 384 px sozinha aparece com bal_acc 0,722 na metade B, contra 0,768
  na validação inteira. Suspeita: o tau escolhido só na metade A ficou ruim para ela (tau da E2 a 384 px
  é sensível). Não afeta o ensemble, mas precisa ser entendido antes de citar esse número.

Finalistas (+2 sementes) iniciados às 02:26: E1, E2 a 384 px, E3 (os 3 maiores bal_acc dos finais).

**02:40 — Anomalia da E2 a 384 px explicada: sensibilidade ao tau.** Na E5, o tau de cada membro é
escolhido só na metade A (~2.900 imagens). Para a E2 a 384 px isso deu **tau 0,25**, contra **0,60** na
validação inteira; com tau 0,60, a bal_acc dela na metade B é **0,779** (não 0,722). A E1 é estável
(tau 0,45 × 0,50; 0,769 × 0,772) e a E3 nem usa o ajuste (treinou com perda ponderada → prior uniforme).
- Leitura: modelos com focal forte + suavização (E2 a 384 px) têm probabilidades achatadas, e a decisão
  em 8 classes passa a depender muito do tau; escolhido numa amostra pequena (classes raras com 20–100
  imagens na metade A), ele varia. A média do ensemble amortece isso.
- Cuidado para os testes: reportar o tau usado e, para os finalistas, conferir a sensibilidade de
  bal_acc ao tau (curva bal_acc × tau na validação). O ensemble com temperatura é a opção mais robusta.

---

## 14/11 06:18 — Finalistas (3 sementes) concluídos

Os 3 maiores bal_acc dos modelos finais (E1, E2 a 384 px, E3) retreinados com as sementes 43 e 44
(mesma configuração; ~4 h de GPU). Média ± desvio-padrão de 3 sementes (validação, TTA 8):

| Modelo | bal_acc | spec95 | AUC mel | AUC mal | ECE |
|---|---|---|---|---|---|
| **E1 ConvNeXt 384** | **0,764 ± 0,005** | **0,766 ± 0,006** | 0,934 ± 0,002 | 0,952 ± 0,001 | 0,314 |
| E3 fusão ConvNeXt+Swin | 0,761 ± 0,003 | 0,754 ± 0,008 | 0,922 ± 0,008 | 0,947 ± 0,002 | 0,207 |
| E2 MaxViT 384 | 0,757 ± 0,010 | 0,746 ± 0,004 | 0,932 ± 0,001 | 0,948 ± 0,001 | 0,273 |

- **bal_acc: empate** (as médias diferem menos que o desvio entre sementes).
- **spec95: vantagem pequena e consistente da E1** (as 3 sementes da E1 ficam acima das 3 da E2 a 384 px).
- Os números de uma semente só eram levemente otimistas (a semente 42 foi a escolhida pela busca):
  E1 0,769 → média 0,764; E2 a 384 px 0,768 → 0,757.

## 14/11 ~06:40 — Notebook regenerado com E4, E5 e finalistas

`notebook_dermatoscopia.ipynb` agora tem seções completas de E4 e E5, a seção dos finalistas (tabela e
gráfico de sementes) e a linha de raciocínio/conclusão atualizadas. Previsões de validação de todos os
modelos (incluindo sementes e o ensemble) em `resultados/previsoes/`.

---

# RESUMO DA 2ª NOITE (13–14/11)

**Tudo concluído, sem falhas nesta noite:** retreino da E3 → E4 (9 tentativas) → E5 → finalistas (6 treinos).

| Etapa | Resultado (validação) | Hipótese |
|---|---|---|
| E0 receita | bal_acc 0,696, spec95 0,689 | régua |
| E0_busca | 0,7345 / 0,703 (+3,9 só com parâmetros) | — |
| E1 ConvNeXt 384 | **0,764 ± 0,005 / 0,766 ± 0,006** (3 sementes); melanoma 0,62 → 0,77 | confirmada |
| E2 MaxViT 224 → 384 | 0,746 → 0,757 ± 0,010; empata com E1 na mesma resolução | não confirmada (confusão bkl/ak/scc igual) |
| E3 fusão CNN+Swin | 0,761 ± 0,003 / 0,754 | não confirmada (não soma à E1) |
| E4 + PanDerm | 0,757 / 0,749 — idêntico à E3 sem PanDerm | não confirmada |
| **E5 ensemble (4 modelos + temperatura)** | metade B: 0,775 / **0,775**, AUC maligno **0,950**, ECE **0,056** | **confirmada** (ordenação e calibração) |

Principais aprendizados: (1) parâmetros e **resolução 384 px** renderam mais que trocar de arquitetura;
(2) ruído entre rodadas ~1 ponto — por isso as 3 sementes; (3) o ensemble é a melhor opção para triagem
(ordena melhor e tem probabilidades confiáveis); (4) ceratose actínica segue a classe mais difícil;
(5) a decisão em 8 classes de modelos com focal forte é sensível ao tau.

Próximo passo (decisão do grupo): abrir os testes T1/T2 uma única vez nos finalistas —
sugestão: **E0 (régua), E1 (3 sementes) e E5** — com `python -m dermato.testar prever ...` e
`relatorio --base E0_final_s42 --finalistas ...` (bootstrap pareado + dermatologistas).

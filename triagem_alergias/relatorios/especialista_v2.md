| teste (802 imagens, 380 casos)     |   top-1 |   top-3 |   bal. acc |   F1 macro |   AUROC |   por caso |
|:-----------------------------------|--------:|--------:|-----------:|-----------:|--------:|-----------:|
| meta_suave_linear                  |   0.266 |   0.63  |      0.128 |      0.128 |   0.732 |      0.268 |
| especialista etapa 4               |   0.281 |   0.688 |      0.176 |      0.189 |   0.808 |      0.332 |
| dinov2_base+caso+meta_unico_linear |   0.329 |   0.585 |      0.293 |      0.258 |   0.798 |      0.329 |
| dinov2_base_suave_linear           |   0.324 |   0.658 |      0.254 |      0.261 |   0.786 |      0.387 |
| dinov2_base+caso+meta_suave_linear |   0.365 |   0.613 |      0.308 |      0.295 |   0.802 |      0.363 |
| dinov2_base+meta_suave_linear      |   0.367 |   0.663 |      0.313 |      0.309 |   0.805 |      0.395 |
| dinov2_base+caso+meta_suave_mlp    |   0.39  |   0.736 |      0.312 |      0.313 |   0.811 |      0.418 |
| etapa 4 + dinov2 mlp (média)       |   0.403 |   0.739 |      0.313 |      0.325 |   0.839 |      0.432 |
"""Busca automática de parâmetros (Optuna) de uma etapa e retreino da configuração vencedora.

Uso:
    python -m dermato.busca E1 [--tentativas 30] [--horas 20] [--rapido]
    python -m dermato.busca E3 --pesos-cnn F:/dermato/resultados/modelos/E1_final_s42.pt
    python -m dermato.busca E4 --pesos-cnn ... --emb F:/dermato/resultados/panderm_emb.npz
    python -m dermato.busca E1 --so-final          # só retreina a melhor configuração já encontrada

Todas as tentativas ficam em resultados/optuna.db (o notebook lê dali a importância dos parâmetros)
e em experimentos.csv. O objetivo é a acurácia balanceada na validação; tentativas ruins são
cortadas a partir da 4ª época (MedianPruner).
"""
import argparse
import json

import optuna

from .config import EXPERIMENTOS, REPO, SEED
from .treinar import treinar

DB = f"sqlite:///{(REPO / 'resultados' / 'optuna.db').as_posix()}"


def comuns(t: optuna.Trial, tem_drop_path=True):
    c = {
        "lr": t.suggest_float("lr", 1e-5, 5e-4, log=True),
        "wd": t.suggest_float("wd", 1e-3, 0.1, log=True),
        "batch": t.suggest_categorical("batch", [16, 32, 64]),
        "epocas": t.suggest_int("epocas", 6, 15),
        "aquecimento": t.suggest_int("aquecimento", 0, 2),
        "perda": t.suggest_categorical("perda", ["ce", "ce_ponderada", "focal"]),
        "suavizacao": t.suggest_float("suavizacao", 0.0, 0.2),
        "aug": t.suggest_categorical("aug", ["leve", "media", "forte"]),
        "mix_p": t.suggest_float("mix_p", 0.0, 0.5),
        "mix_tipo": t.suggest_categorical("mix_tipo", ["mixup", "cutmix"]),
        "sog": t.suggest_categorical("sog", [True, False]),
        "dropout": t.suggest_float("dropout", 0.0, 0.5),
        "bn_congelada": t.suggest_categorical("bn_congelada", [True, False]),
        "meta": t.suggest_categorical("meta", [True, False]),
        "max_nevos": t.suggest_categorical("max_nevos", [3000, 6000, 0]),   # 0 = sem limite
        "balanceado": t.suggest_categorical("balanceado", [True, False]),
        "paciencia": 3,
    }
    if c["perda"] == "focal":
        c["gamma"] = t.suggest_float("gamma", 0.5, 3.0)
    if tem_drop_path:
        c["drop_path"] = t.suggest_float("drop_path", 0.0, 0.3)
    return c


def refinados(t: optuna.Trial, epocas=(8, 15)):
    """Espaço estreitado em volta do que funcionou na E0_busca, E1 e E2 (ver DIARIO.md, 13/11):
    fora lr < 1,5e-4 (sempre ruim), augmentation forte, SoG, amostragem balanceada e BN treinável
    (nenhuma das melhores usou); perda focal com gama até 5 (na E1 o gama ficou no teto, 2,97 de 3)."""
    return {
        "lr": t.suggest_float("lr", 1.5e-4, 4e-4, log=True),
        "wd": t.suggest_float("wd", 0.005, 0.05, log=True),
        "batch": t.suggest_categorical("batch", [16, 32]),
        "epocas": t.suggest_int("epocas", *epocas),
        "aquecimento": t.suggest_int("aquecimento", 0, 1),
        "perda": "focal",
        "gamma": t.suggest_float("gamma", 2.0, 5.0),
        "suavizacao": t.suggest_float("suavizacao", 0.05, 0.2),
        "aug": t.suggest_categorical("aug", ["leve", "media"]),
        "mix_p": t.suggest_float("mix_p", 0.0, 0.4),
        "mix_tipo": t.suggest_categorical("mix_tipo", ["mixup", "cutmix"]),
        "sog": False,
        "dropout": t.suggest_float("dropout", 0.1, 0.5),
        "drop_path": t.suggest_float("drop_path", 0.0, 0.2),
        "bn_congelada": True,
        "meta": True,
        "max_nevos": t.suggest_categorical("max_nevos", [0, 6000]),
        "balanceado": False,
        "cabeca_epocas": t.suggest_int("cabeca_epocas", 0, 1),
        "paciencia": 3,
    }


def espaco(etapa, args):
    def fn(t: optuna.Trial):
        if etapa == "E0_busca":
            # mesma rede do E0 (EfficientNetV2-B0), agora com todos os parâmetros livres
            c = {"backbone": "effv2_b0", "res": t.suggest_categorical("res", [224, 288, 384])}
            c |= comuns(t)
            c["cabeca_epocas"] = t.suggest_int("cabeca_epocas", 0, 2)
        elif etapa == "E1":
            c = {"backbone": "convnext_tiny", "res": t.suggest_categorical("res", [224, 288, 384])}
            c |= comuns(t)
            c["cabeca_epocas"] = t.suggest_int("cabeca_epocas", 0, 2)
        elif etapa == "E2":
            # 384 px custa 4 min/época na RTX 5070: fica em 224, como o maxvit_tiny_tf_224 do plano
            c = {"backbone": "maxvit_tiny_224", "res": 224}
            c |= comuns(t)
            c["cabeca_epocas"] = t.suggest_int("cabeca_epocas", 0, 2)
            c["layer_decay"] = t.suggest_float("layer_decay", 0.6, 1.0)
        elif etapa == "E1r":
            # E1 refinada: ConvNeXt-Tiny a 384 ou 448 px (448 = limite do cache)
            c = {"backbone": "convnext_tiny", "res": t.suggest_categorical("res", [384, 448])}
            c |= refinados(t)
        elif etapa == "E2_384":
            # E2 na mesma resolução da E1, para separar arquitetura de resolução (~1 h por tentativa)
            c = {"backbone": "maxvit_tiny_384", "res": 384, "layer_decay": 0.8}   # decaimento não mudou nada na E2
            c |= refinados(t, epocas=(6, 12))
        elif etapa in ("E3", "E4"):
            c = {"modelo": "fusao", "res": t.suggest_categorical("res", [224, 288, 384]),
                 "pesos_cnn": args.pesos_cnn}
            c |= comuns(t)
            c["juncao"] = t.suggest_categorical("juncao", ["concat", "porta", "atencao"])
            c["lr_vit_mult"] = t.suggest_float("lr_vit_mult", 0.1, 1.0, log=True)
            c["ramos_congelados_epocas"] = t.suggest_int("ramos_congelados_epocas", 0, 3)
            c["hidden"] = t.suggest_categorical("hidden", [256, 512])
            if etapa == "E4":
                c["emb"] = args.emb
                c["emb_dropout"] = t.suggest_float("emb_dropout", 0.0, 0.6)
                c["emb_norm"] = t.suggest_categorical("emb_norm", [True, False])
        else:
            raise ValueError(etapa)
        c |= {"etapa": etapa, "seed": SEED, "rapido": args.rapido}
        return c
    return fn


def objetivo(etapa, args):
    montar = espaco(etapa, args)

    def obj(t):
        cfg = montar(t)
        t.set_user_attr("cfg", json.dumps(cfg, default=str))
        r = treinar(cfg, trial=t)
        for k in ("val_spec95", "val_mel_auc", "val_mal_auc", "val_ece", "tempo_treino_min"):
            t.set_user_attr(k, r[k])
        return r["val_bal_acc"]
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["E0_busca", "E1", "E1r", "E2", "E2_384", "E3", "E4"])
    ap.add_argument("--tentativas", type=int, default=30)
    ap.add_argument("--horas", type=float, default=None, help="limite de tempo da busca")
    ap.add_argument("--pesos-cnn", help="modelo final da E1 (.pt), ponto de partida do ramo ConvNeXt")
    ap.add_argument("--emb", help="vetores do PanDerm (.npz), para a E4")
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--so-final", action="store_true")
    args = ap.parse_args()
    if args.etapa in ("E3", "E4") and not args.pesos_cnn:
        ap.error("E3/E4 precisam de --pesos-cnn (modelo final da E1)")
    if args.etapa == "E4" and not args.emb:
        ap.error("E4 precisa de --emb")

    nome = args.etapa + ("_rapido" if args.rapido else "")
    # semente do sorteador = 42 + nº de tentativas já no estudo: ao retomar uma busca interrompida,
    # o TPE não repete as mesmas configurações iniciais (aconteceu na E0_busca, tentativas 0 e 2)
    try:
        ja = len(optuna.load_study(study_name=nome, storage=DB).trials)
    except KeyError:
        ja = 0
    estudo = optuna.create_study(
        study_name=nome, storage=DB, load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED + ja, multivariate=True),
        # nas buscas refinadas (semeadas com configurações boas) o corte pode começar mais cedo
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3 if args.etapa in ("E1r", "E2_384") else 5,
                                           n_warmup_steps=3))
    if not args.so_final:
        feitas = len([t for t in estudo.trials if t.state.is_finished()])
        faltam = max(0, args.tentativas - feitas)
        print(f"Busca {nome}: {feitas} tentativas já feitas, rodando mais {faltam}", flush=True)
        estudo.optimize(objetivo(args.etapa, args), n_trials=faltam,
                        timeout=args.horas * 3600 if args.horas else None, catch=(RuntimeError,))

    melhor = estudo.best_trial
    if "cfg" in melhor.user_attrs:
        cfg = json.loads(melhor.user_attrs["cfg"])
    else:   # tentativa registrada sem treino (resultado já conhecido): reconstrói a config dos parâmetros
        cfg = espaco(args.etapa, args)(optuna.trial.FixedTrial(melhor.params))
    print(f"Melhor tentativa: #{melhor.number} val_bal_acc {melhor.value:.4f}\n{json.dumps(cfg, indent=1)}")
    # retreino da vencedora com todas as épocas (sem corte), salvando o modelo
    final = treinar(cfg | {"salvar_modelo": True, "paciencia": 99}, nome=f"{nome}_final_s{SEED}")
    print(f"Modelo final: {final.get('modelo')}  (experimentos em {EXPERIMENTOS})")


if __name__ == "__main__":
    main()

"""Velocidade de treino (img/s) e memória de cada rede na GPU, com dados sintéticos."""
import time

import torch
import torch.nn.functional as F

from .modelos import Fusao, Rede


def medir(nome, model, res, batch, passos=15):
    model = model.cuda().to(memory_format=torch.channels_last).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(batch, 3, res, res, device="cuda").to(memory_format=torch.channels_last)
    m = torch.randn(batch, 14, device="cuda")
    y = torch.randint(0, 8, (batch,), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    try:
        for k in range(passos + 3):
            if k == 3:
                torch.cuda.synchronize()
                t = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x, m).float(), y)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        ips = passos * batch / (time.time() - t)
        print(f"{nome:28s} {res}px lote {batch:3d}: {ips:6.0f} img/s | {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
    except torch.OutOfMemoryError:
        print(f"{nome:28s} {res}px lote {batch:3d}: SEM MEMÓRIA", flush=True)
    del model, opt
    torch.cuda.empty_cache()


if __name__ == "__main__":
    medir("effv2_b0", Rede("effv2_b0"), 224, 64)
    for res, b in [(224, 64), (288, 48), (384, 32)]:
        medir("convnext_tiny", Rede("convnext_tiny"), res, b)
    medir("maxvit_tiny_224", Rede("maxvit_tiny_224"), 224, 32)
    medir("maxvit_tiny_384", Rede("maxvit_tiny_384"), 384, 16)
    for res, b in [(224, 32), (384, 24)]:
        medir("fusao convnext+swin", Fusao("atencao"), res, b)

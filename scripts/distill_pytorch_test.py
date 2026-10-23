import os

import pytest
import torch

os.environ["JAX_PLATFORMS"] = "cpu"

from . import distill_pytorch

DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def _adamw(params):
    # Phase-2 settings: peak LR 2.5e-5, openpi's AdamW betas.
    return torch.optim.AdamW(params, lr=2.5e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)


@pytest.mark.parametrize("device", DEVICES)
def test_pure_bf16_adamw_drops_small_updates(device: str):
    # The phase-2 failure: half a bf16 ulp at |w| = 0.05 is 1.2e-4 > lr, so every AdamW step rounds away.
    w = torch.nn.Parameter(torch.full((256,), 0.05, dtype=torch.bfloat16, device=device))
    init = w.detach().clone()
    opt = _adamw([w])
    for _ in range(100):
        w.grad = torch.ones_like(w)
        opt.step()
        opt.zero_grad(set_to_none=True)
    assert torch.equal(w.detach(), init)


@pytest.mark.parametrize("device", DEVICES)
def test_fp32_master_weights_keep_small_updates(device: str):
    w = torch.nn.Parameter(torch.full((256,), 0.05, dtype=torch.bfloat16, device=device))
    # Mixed list as in the real student (bf16 matrices + fp32 adaRMS/projections): the copy-back must handle both.
    f = torch.nn.Parameter(torch.full((3,), 0.5, dtype=torch.float32, device=device))
    params = [w, f]
    masters = distill_pytorch.init_fp32_master_weights(params)
    opt = _adamw(masters)
    for _ in range(100):
        w.grad = torch.ones_like(w)
        f.grad = torch.ones_like(f)
        distill_pytorch.fp32_master_weight_step(opt, params, masters)
    # 100 steps of ~lr each: about -2.5e-3, which is visible at bf16 resolution.
    assert torch.allclose(w.detach().float(), torch.full_like(w, 0.05 - 100 * 2.5e-5, dtype=torch.float32), atol=4e-4)
    assert torch.allclose(f.detach(), torch.full_like(f, 0.5 - 100 * 2.5e-5), atol=1e-5)
    assert w.dtype == torch.bfloat16
    assert f.dtype == torch.float32
    assert w.grad is None
    assert f.grad is None

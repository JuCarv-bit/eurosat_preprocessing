import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from simclr.config import CONFIG

def make_optimizer_scheduler(params, lr, wd, steps_per_epoch, epochs):
    total_steps  = epochs * steps_per_epoch
    warmup_steps = steps_per_epoch
    opt = optim.AdamW(params, lr=lr, betas=(0.9,0.98), eps=1e-8, weight_decay=wd)
    sched = SequentialLR(
        opt,
        schedulers=[
            LinearLR(opt,  start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(opt, T_max=total_steps - warmup_steps)
        ],
        milestones=[warmup_steps]
    )
    return opt, sched
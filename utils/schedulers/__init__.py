from torch.optim.lr_scheduler import CosineAnnealingLR, ConstantLR

from .tristage_scheduler import TriStageLRScheduler


def fetch_scheduler(type_, optimizer, train_iters):
    if type_ == "constant":
        scheduler = ConstantLR(optimizer, factor=1.0, total_iters=train_iters)
    elif type_ == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=train_iters)
    elif type_ == "tristage_flower":
        scheduler = TriStageLRScheduler(optimizer)
    else:
        raise NotImplementedError

    return scheduler

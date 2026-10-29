import random

import numpy as np
import torch


def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

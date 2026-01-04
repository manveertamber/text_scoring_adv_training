import os, random
import torch
import transformers
import numpy as np

def seed_everything(seed: int, add_rank: bool = True) -> int:
    if add_rank and torch.distributed.is_available() and torch.distributed.is_initialized():
        seed = int(seed) + int(torch.distributed.get_rank())

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    transformers.set_seed(seed)

    return seed

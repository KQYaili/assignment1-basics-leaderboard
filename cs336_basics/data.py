import numpy as np
import torch


def get_batch(dataset, batch_size, context_length, device):
    i=np.random.randint(0, len(dataset) - context_length, size=batch_size)
    x=dataset[i[:,None]+np.arange(context_length)]
    y=dataset[(i+1)[:,None]+np.arange(context_length)]
    return torch.tensor(x, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)

import torch
from torch.distributions import Categorical


def normalized_entropy_confidence(probs):
    normalized_entropy = Categorical(probs = probs).entropy() / torch.log(torch.tensor(probs.shape[-1]))
    return 1.0 - normalized_entropy
